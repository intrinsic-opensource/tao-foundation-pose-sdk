/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "foundation_pose_nvidia/mesh.hpp"
#include "foundation_pose_nvidia/exception.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <png.h>

#include "math_utils.hpp"

namespace foundation_pose_nvidia {
// Vec3f arithmetic operators are hidden friends of Vec3f (types.hpp), found via
// argument-dependent lookup.
namespace {

std::string lowerExtension(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::ranges::transform(ext, ext.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

void loadPngTexture(const std::filesystem::path& path, Mesh& mesh) {
  FILE* file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    return;
  }
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (png == nullptr) {
    std::fclose(file);
    return;
  }
  png_infop info = png_create_info_struct(png);
  if (info == nullptr) {
    png_destroy_read_struct(&png, nullptr, nullptr);
    std::fclose(file);
    return;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(file);
    mesh.texture_width = 0;
    mesh.texture_height = 0;
    mesh.texture_rgb.clear();
    return;
  }

  png_init_io(png, file);
  png_read_info(png, info);

  png_uint_32 width = 0;
  png_uint_32 height = 0;
  int bit_depth = 0;
  int color_type = 0;
  png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr,
               nullptr, nullptr);

  if (bit_depth == 16) {
    png_set_strip_16(png);
  }
  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png);
  }
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png);
  }
  if (png_get_valid(png, info, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png);
  }
  if (color_type == PNG_COLOR_TYPE_GRAY ||
      color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
    png_set_gray_to_rgb(png);
  }
  if (color_type & PNG_COLOR_MASK_ALPHA) {
    png_set_strip_alpha(png);
  }

  png_read_update_info(png, info);
  const png_size_t row_bytes = png_get_rowbytes(png, info);
  std::vector<std::uint8_t> raw(row_bytes * height);
  std::vector<png_bytep> rows(height);
  for (png_uint_32 y = 0; y < height; ++y) {
    rows[y] = raw.data() + y * row_bytes;
  }
  png_read_image(png, rows.data());
  png_destroy_read_struct(&png, &info, nullptr);
  std::fclose(file);

  mesh.texture_width = static_cast<int>(width);
  mesh.texture_height = static_cast<int>(height);
  mesh.texture_rgb.assign(static_cast<std::size_t>(width * height), Vec3u8{});
  for (png_uint_32 y = 0; y < height; ++y) {
    for (png_uint_32 x = 0; x < width; ++x) {
      const std::uint8_t* src = raw.data() + y * row_bytes + x * 3;
      mesh.texture_rgb[static_cast<std::size_t>(y * width + x)] =
          Vec3u8{src[0], src[1], src[2]};
    }
  }
}

void stripTrailingCr(std::string& line) {
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
}

std::uint32_t resolveIndex(int idx, std::size_t count) {
  if (idx > 0) {
    return static_cast<std::uint32_t>(idx - 1);
  }
  if (idx < 0) {
    return static_cast<std::uint32_t>(static_cast<int>(count) + idx);
  }
  throw FoundationPoseError("OBJ indices are 1-based; got index 0");
}

std::uint32_t parseObjVertexIndex(std::string_view token, std::size_t vertex_count) {
  const std::size_t slash = token.find('/');
  const std::string head(token.substr(0, slash));
  return resolveIndex(std::stoi(head), vertex_count);
}

Mesh loadObj(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw FoundationPoseError("Failed to open OBJ mesh: " + path.string());
  }

  Mesh mesh;
  mesh.source_path = path.string();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    std::string tag;
    iss >> tag;
    if (tag == "v") {
      Vec3f p{};
      iss >> p.x >> p.y >> p.z;
      mesh.vertices.push_back(p);
      float r = 128.0f;
      float g = 128.0f;
      float b = 128.0f;
      if ((iss >> r >> g >> b) && r <= 1.0f && g <= 1.0f && b <= 1.0f) {
        r *= 255.0f;
        g *= 255.0f;
        b *= 255.0f;
      }
      mesh.vertex_colors.push_back(Vec3u8{
          static_cast<std::uint8_t>(std::clamp(r, 0.0f, 255.0f)),
          static_cast<std::uint8_t>(std::clamp(g, 0.0f, 255.0f)),
          static_cast<std::uint8_t>(std::clamp(b, 0.0f, 255.0f)),
      });
    } else if (tag == "vt") {
      Vec2f uv{};
      iss >> uv.x >> uv.y;
      mesh.uvs.push_back(uv);
    } else if (tag == "f") {
      std::vector<std::uint32_t> indices;
      std::string token;
      while (iss >> token) {
        indices.push_back(parseObjVertexIndex(token, mesh.vertices.size()));
      }
      if (indices.size() < 3) {
        continue;
      }
      for (std::size_t i = 1; i + 1 < indices.size(); ++i) {
        mesh.faces.push_back({indices[0], indices[i], indices[i + 1]});
      }
    }
  }

  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw FoundationPoseError("OBJ mesh has no usable vertices/faces: " + path.string());
  }
  computeVertexNormals(mesh);
  return mesh;
}

enum class PlyFormat {
  Ascii,
  BinaryLittleEndian,
};

struct PlyProperty {
  bool is_list = false;
  std::string type;
  std::string count_type;
  std::string item_type;
  std::string name;
};

struct PlyElement {
  std::string name;
  std::size_t count = 0;
  std::vector<PlyProperty> properties;
};

template <class T>
T readBinary(std::istream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw FoundationPoseError("Unexpected EOF while reading binary PLY");
  }
  return value;
}

double readBinaryScalar(std::istream& in, const std::string& type) {
  if (type == "char" || type == "int8") {
    return readBinary<std::int8_t>(in);
  }
  if (type == "uchar" || type == "uint8") {
    return readBinary<std::uint8_t>(in);
  }
  if (type == "short" || type == "int16") {
    return readBinary<std::int16_t>(in);
  }
  if (type == "ushort" || type == "uint16") {
    return readBinary<std::uint16_t>(in);
  }
  if (type == "int" || type == "int32") {
    return readBinary<std::int32_t>(in);
  }
  if (type == "uint" || type == "uint32") {
    return readBinary<std::uint32_t>(in);
  }
  if (type == "float" || type == "float32") {
    return readBinary<float>(in);
  }
  if (type == "double" || type == "float64") {
    return readBinary<double>(in);
  }
  throw FoundationPoseError("Unsupported PLY scalar type: " + type);
}

std::uint32_t readBinaryCount(std::istream& in, const std::string& type) {
  const double value = readBinaryScalar(in, type);
  if (value < 0.0) {
    throw FoundationPoseError("Negative PLY list count");
  }
  return static_cast<std::uint32_t>(value);
}

double parseAsciiScalar(std::istringstream& iss, const std::string&) {
  double value = 0.0;
  iss >> value;
  if (!iss) {
    throw FoundationPoseError("Malformed ASCII PLY scalar");
  }
  return value;
}

void assignVertexProperty(Mesh& mesh,
                          std::size_t index,
                          std::string_view name,
                          double value,
                          Vec3f& normal,
                          bool& has_normal) {
  Vec3f& v = mesh.vertices[index];
  if (name == "x") {
    v.x = static_cast<float>(value);
  } else if (name == "y") {
    v.y = static_cast<float>(value);
  } else if (name == "z") {
    v.z = static_cast<float>(value);
  } else if (name == "nx") {
    normal.x = static_cast<float>(value);
    has_normal = true;
  } else if (name == "ny") {
    normal.y = static_cast<float>(value);
    has_normal = true;
  } else if (name == "nz") {
    normal.z = static_cast<float>(value);
    has_normal = true;
  } else if (name == "red" || name == "diffuse_red") {
    mesh.vertex_colors[index].r =
        static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
  } else if (name == "green" || name == "diffuse_green") {
    mesh.vertex_colors[index].g =
        static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
  } else if (name == "blue" || name == "diffuse_blue") {
    mesh.vertex_colors[index].b =
        static_cast<std::uint8_t>(std::clamp(value, 0.0, 255.0));
  }
}

struct PlyHeader {
  PlyFormat format = PlyFormat::Ascii;
  std::string texture_file;
  std::vector<PlyElement> elements;
};

PlyFormat parsePlyFormat(std::istringstream& iss) {
  std::string fmt;
  iss >> fmt;
  if (fmt == "ascii") {
    return PlyFormat::Ascii;
  }
  if (fmt == "binary_little_endian") {
    return PlyFormat::BinaryLittleEndian;
  }
  throw FoundationPoseError("Unsupported PLY format: " + fmt);
}

PlyProperty parsePlyProperty(std::istringstream& iss) {
  PlyProperty prop;
  std::string kind;
  iss >> kind;
  if (kind == "list") {
    prop.is_list = true;
    iss >> prop.count_type >> prop.item_type >> prop.name;
  } else {
    prop.type = kind;
    iss >> prop.name;
  }
  return prop;
}

// Parses the PLY header (magic line, format, texture comment, element/property
// declarations) up to and including "end_header".
PlyHeader parsePlyHeader(std::istream& in, const std::filesystem::path& path) {
  std::string line;
  std::getline(in, line);
  stripTrailingCr(line);
  if (line != "ply") {
    throw FoundationPoseError("Not a PLY file: " + path.string());
  }

  PlyHeader header;
  PlyElement* current = nullptr;
  while (std::getline(in, line)) {
    stripTrailingCr(line);
    if (line == "end_header") {
      break;
    }
    std::istringstream iss(line);
    std::string tag;
    iss >> tag;
    if (tag == "comment") {
      std::string key;
      iss >> key;
      if (key == "TextureFile") {
        iss >> header.texture_file;
      }
    } else if (tag == "format") {
      header.format = parsePlyFormat(iss);
    } else if (tag == "element") {
      PlyElement element;
      iss >> element.name >> element.count;
      header.elements.push_back(element);
      current = &header.elements.back();
    } else if (tag == "property" && current != nullptr) {
      current->properties.push_back(parsePlyProperty(iss));
    }
  }
  return header;
}

// Reads one vertex's properties from a scalar/count source (an ASCII line's
// stream or the binary stream) and stores position/color/normal/uv into the
// mesh. read_scalar returns the next scalar (as double); read_count returns the
// next list length.
template <class ReadScalar, class ReadCount>
void readPlyVertex(const PlyElement& element, std::size_t index, Mesh& mesh,
                   ReadScalar read_scalar, ReadCount read_count) {
  Vec3f normal{};
  bool has_normal = false;
  float texture_u = 0.0f;
  float texture_v = 0.0f;
  bool has_texture_uv = false;
  for (const PlyProperty& prop : element.properties) {
    if (prop.is_list) {
      const std::uint32_t count = read_count(prop.count_type);
      for (std::uint32_t j = 0; j < count; ++j) {
        (void)read_scalar(prop.item_type);
      }
    } else {
      const double value = read_scalar(prop.type);
      if (prop.name == "texture_u") {
        texture_u = static_cast<float>(value);
        has_texture_uv = true;
      } else if (prop.name == "texture_v") {
        texture_v = static_cast<float>(value);
        has_texture_uv = true;
      } else {
        assignVertexProperty(mesh, index, prop.name, value, normal, has_normal);
      }
    }
  }
  if (has_texture_uv) {
    if (mesh.uvs.size() != mesh.vertices.size()) {
      mesh.uvs.resize(mesh.vertices.size());
    }
    mesh.uvs[index] = {texture_u, texture_v};
  }
  if (has_normal) {
    mesh.normals[index] = detail::normalize(normal);
  }
}

// Reads one face's vertex-index list (ignoring other properties) and appends
// its triangle fan to the mesh.
template <class ReadScalar, class ReadCount>
void readPlyFace(const PlyElement& element, Mesh& mesh, ReadScalar read_scalar,
                 ReadCount read_count) {
  std::vector<std::uint32_t> indices;
  for (const PlyProperty& prop : element.properties) {
    if (prop.is_list) {
      const std::uint32_t count = read_count(prop.count_type);
      std::vector<std::uint32_t> values(count);
      for (std::uint32_t j = 0; j < count; ++j) {
        values[j] = static_cast<std::uint32_t>(read_scalar(prop.item_type));
      }
      if (prop.name == "vertex_indices" || prop.name == "vertex_index") {
        indices = std::move(values);
      }
    } else {
      (void)read_scalar(prop.type);
    }
  }
  if (indices.size() >= 3) {
    for (std::size_t j = 1; j + 1 < indices.size(); ++j) {
      mesh.faces.push_back({indices[0], indices[j], indices[j + 1]});
    }
  }
}

// Lambda factories for the ASCII (per-line istringstream) and binary (file
// stream) scalar/count sources, so the element readers stay agnostic to format.
auto asciiScalarReader(std::istringstream& iss) {
  return [&iss](const std::string& type) { return parseAsciiScalar(iss, type); };
}
auto asciiCountReader(std::istringstream& iss) {
  return [&iss](const std::string& type) {
    return static_cast<std::uint32_t>(parseAsciiScalar(iss, type));
  };
}
auto binaryScalarReader(std::istream& in) {
  return [&in](const std::string& type) { return readBinaryScalar(in, type); };
}
auto binaryCountReader(std::istream& in) {
  return [&in](const std::string& type) { return readBinaryCount(in, type); };
}

void readVertexElement(std::istream& in, PlyFormat format,
                       const PlyElement& element, Mesh& mesh) {
  mesh.vertices.resize(element.count);
  mesh.vertex_colors.assign(element.count, Vec3u8{128, 128, 128});
  mesh.normals.resize(element.count);
  std::string line;
  for (std::size_t i = 0; i < element.count; ++i) {
    if (format == PlyFormat::Ascii) {
      std::getline(in, line);
      stripTrailingCr(line);
      std::istringstream iss(line);
      readPlyVertex(element, i, mesh, asciiScalarReader(iss), asciiCountReader(iss));
    } else {
      readPlyVertex(element, i, mesh, binaryScalarReader(in), binaryCountReader(in));
    }
  }
}

void readFaceElement(std::istream& in, PlyFormat format,
                     const PlyElement& element, Mesh& mesh) {
  std::string line;
  for (std::size_t i = 0; i < element.count; ++i) {
    if (format == PlyFormat::Ascii) {
      std::getline(in, line);
      stripTrailingCr(line);
      std::istringstream iss(line);
      readPlyFace(element, mesh, asciiScalarReader(iss), asciiCountReader(iss));
    } else {
      readPlyFace(element, mesh, binaryScalarReader(in), binaryCountReader(in));
    }
  }
}

// Consumes an unsupported element's binary payload to keep the stream aligned.
void skipBinaryProperties(std::istream& in, const PlyElement& element) {
  for (const PlyProperty& prop : element.properties) {
    if (prop.is_list) {
      const std::uint32_t count = readBinaryCount(in, prop.count_type);
      for (std::uint32_t j = 0; j < count; ++j) {
        (void)readBinaryScalar(in, prop.item_type);
      }
    } else {
      (void)readBinaryScalar(in, prop.type);
    }
  }
}

void skipPlyElement(std::istream& in, PlyFormat format,
                    const PlyElement& element) {
  std::string line;
  for (std::size_t i = 0; i < element.count; ++i) {
    if (format == PlyFormat::Ascii) {
      std::getline(in, line);
    } else {
      skipBinaryProperties(in, element);
    }
  }
}

Mesh loadPly(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw FoundationPoseError("Failed to open PLY mesh: " + path.string());
  }
  const PlyHeader header = parsePlyHeader(in, path);

  Mesh mesh;
  mesh.source_path = path.string();
  for (const PlyElement& element : header.elements) {
    if (element.name == "vertex") {
      readVertexElement(in, header.format, element, mesh);
    } else if (element.name == "face") {
      readFaceElement(in, header.format, element, mesh);
    } else {
      skipPlyElement(in, header.format, element);
    }
  }

  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw FoundationPoseError("PLY mesh has no usable vertices/faces: " + path.string());
  }
  if (const bool normals_valid =
          mesh.normals.size() == mesh.vertices.size() &&
          std::ranges::any_of(mesh.normals,
                              [](Vec3f n) { return detail::norm(n) > 0.5f; });
      !normals_valid) {
    computeVertexNormals(mesh);
  }
  if (!header.texture_file.empty()) {
    loadPngTexture(path.parent_path() / header.texture_file, mesh);
  }
  return mesh;
}

struct VoxelKey {
  long long x = 0;
  long long y = 0;
  long long z = 0;

  bool operator==(const VoxelKey& other) const = default;
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& k) const {
    const std::size_t h1 = std::hash<long long>{}(k.x);
    const std::size_t h2 = std::hash<long long>{}(k.y);
    const std::size_t h3 = std::hash<long long>{}(k.z);
    return h1 ^ (h2 << 1U) ^ (h3 << 2U);
  }
};

}  // namespace

Mesh loadMesh(const std::filesystem::path& path) {
  const std::string ext = lowerExtension(path);
  if (ext == ".obj") {
    return loadObj(path);
  }
  if (ext == ".ply") {
    return loadPly(path);
  }
  throw FoundationPoseError("Unsupported mesh extension '" + ext +
                           "'. Expected .obj or .ply");
}

void computeVertexNormals(Mesh& mesh) {
  mesh.normals.assign(mesh.vertices.size(), Vec3f{});
  for (const auto& f : mesh.faces) {
    if (f[0] >= mesh.vertices.size() || f[1] >= mesh.vertices.size() ||
        f[2] >= mesh.vertices.size()) {
      continue;
    }
    const Vec3f a = mesh.vertices[f[0]];
    const Vec3f b = mesh.vertices[f[1]];
    const Vec3f c = mesh.vertices[f[2]];
    const Vec3f n = detail::cross(b - a, c - a);
    mesh.normals[f[0]] = mesh.normals[f[0]] + n;
    mesh.normals[f[1]] = mesh.normals[f[1]] + n;
    mesh.normals[f[2]] = mesh.normals[f[2]] + n;
  }
  for (Vec3f& n : mesh.normals) {
    n = detail::normalize(n);
    if (detail::norm(n) < 1e-6f) {
      n = {0.0f, 0.0f, 1.0f};
    }
  }
}

PreprocessedMesh preprocessMesh(const Mesh& mesh, const Config& config) {
  if (mesh.vertices.size() < 4 || mesh.faces.empty()) {
    throw FoundationPoseError("Mesh must contain at least four vertices and one face");
  }

  Vec3f min_v{std::numeric_limits<float>::max(),
              std::numeric_limits<float>::max(),
              std::numeric_limits<float>::max()};
  Vec3f max_v{std::numeric_limits<float>::lowest(),
              std::numeric_limits<float>::lowest(),
              std::numeric_limits<float>::lowest()};
  for (Vec3f v : mesh.vertices) {
    min_v.x = std::min(min_v.x, v.x);
    min_v.y = std::min(min_v.y, v.y);
    min_v.z = std::min(min_v.z, v.z);
    max_v.x = std::max(max_v.x, v.x);
    max_v.y = std::max(max_v.y, v.y);
    max_v.z = std::max(max_v.z, v.z);
  }

  PreprocessedMesh out;
  out.center = (min_v + max_v) * 0.5f;
  out.centered_mesh = mesh;
  for (Vec3f& v : out.centered_mesh.vertices) {
    v = v - out.center;
  }

  const std::size_t sample_count =
      std::min<std::size_t>(static_cast<std::size_t>(config.mesh_sample_points),
                            out.centered_mesh.vertices.size());
  const std::size_t stride =
      std::max<std::size_t>(1, out.centered_mesh.vertices.size() / sample_count);
  std::vector<Vec3f> sampled;
  sampled.reserve(sample_count);
  for (std::size_t i = 0; i < out.centered_mesh.vertices.size() &&
                          sampled.size() < sample_count;
       i += stride) {
    sampled.push_back(out.centered_mesh.vertices[i]);
  }

  float diameter = 0.0f;
  for (std::size_t i = 0; i < sampled.size(); ++i) {
    for (std::size_t j = i + 1; j < sampled.size(); ++j) {
      diameter = std::max(diameter, detail::norm(sampled[i] - sampled[j]));
    }
  }
  if (diameter <= 0.0f) {
    diameter = detail::norm(max_v - min_v);
  }
  out.diameter = std::max(diameter, 1e-6f);

  const float voxel_size =
      std::max(out.diameter / config.voxel_downsample_ratio, config.min_voxel_size);
  std::unordered_map<VoxelKey, Vec3f, VoxelKeyHash> voxels;
  for (Vec3f p : out.centered_mesh.vertices) {
    const VoxelKey key{
        static_cast<long long>(std::floor(p.x / voxel_size)),
        static_cast<long long>(std::floor(p.y / voxel_size)),
        static_cast<long long>(std::floor(p.z / voxel_size)),
    };
    voxels.try_emplace(key, p);
  }
  out.downsampled_points.reserve(voxels.size());
  for (const auto& [_, p] : voxels) {
    out.downsampled_points.push_back(p);
  }

  return out;
}

}  // namespace foundation_pose_nvidia
