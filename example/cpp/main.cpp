// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// sample_app_cpp — single-object FoundationPose C API, two modes.
//
// Without --use_synthesize (default):
//   Reads real RGB-D frames from the BOP YCB-V dataset (scene 000048).
//   Requires data/BOP_datasets/ycbv/test/000048/ to be present.
//   Download with:  scripts/download_bop_ycbv.sh
//
// With --use_synthesize:
//   Generates synthetic RGB-D frames in code (no dataset needed).
//
// In both modes the same fp_* lifecycle is exercised:
//   fp_create -> fp_prepare -> fp_register_frame -> fp_track_frame (x4) -> fp_destroy
//
// Per-frame output (no CSV, no timing):
//   frame N/5  register  score=  X.XXX  pose: [r00 r01 r02 tx / r10 r11 r12 ty / r20 r21 r22 tz]

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

#include "foundation_pose_nvidia/c_api.h"
#include "foundation_pose_nvidia/exception.hpp"
#include "frame_data.hpp"

namespace {

// Collect image IDs from the BOP rgb/ directory (stem parsed as int), sorted.
// Returns at most `max_n` IDs; throws if fewer than max_n exist.
std::vector<int> collectFrameIds(const std::string& rgb_dir, int max_n) {
  std::vector<int> ids;
  for (auto& entry : std::filesystem::directory_iterator(rgb_dir)) {
    if (entry.path().extension() == ".png") {
      ids.push_back(std::stoi(entry.path().stem().string()));
    }
  }
  std::ranges::sort(ids);
  if (static_cast<int>(ids.size()) < max_n) {
    throw foundation_pose_nvidia::FoundationPoseError(
        std::format("expected {} frames in {} but found only {}", max_n,
                    rgb_dir, ids.size()));
  }
  ids.resize(max_n);
  return ids;
}

// ---- Scenario constants ----------------------------------------------------
constexpr int NUMBER_OF_FRAMES = 5;  // 1 register + 4 track
constexpr int kSyntheticWidth = 640;
constexpr int kSyntheticHeight = 480;

// PLY is in mm -> scale to meters.
constexpr float MESH_UNIT_SCALE = 0.001f;

// Register hypothesis count. FP_N_HYPOTHESES overrides the library default
// (252, benchmark grade); lower it (e.g. 12) to shrink the register workspace
// and TensorRT context when the GPU is busy or small.
int nHypotheses() {
  const char* env = std::getenv("FP_N_HYPOTHESES");
  const int value = env != nullptr ? std::atoi(env) : 0;
  return value > 0 ? value : 252;
}

// ---- Paths -----------------------------------------------------------------
const std::string kMeshPath     = "data/BOP_datasets/ycbv/models/obj_000001.ply";
const std::string kBopDir       = "data/BOP_datasets/ycbv/test/000048";
const std::string kRefineModel  = "/work/weights/refiner_net.onnx";
const std::string kScoreModel   = "/work/weights/score_net.onnx";
const std::string kEngineCache  = "/work/engine_cache";

// ---- Camera intrinsics for BOP YCB-V scene 000048 -------------------------
//   K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]  (3x3 row-major)
constexpr float CAMERA_K[9] = {1066.778f, 0.0f, 312.9869f,
                               0.0f,      1067.487f, 241.3109f,
                               0.0f,      0.0f,      1.0f};
constexpr float DEPTH_SCALE = 0.1f;  // depth_png_uint16 * DEPTH_SCALE / 1000 = meters

// ---- Synthetic frame -------------------------------------------------------
struct Frame {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;   // HWC, width*height*3
  std::vector<float> depth;        // HW, meters
  std::vector<std::uint8_t> mask;  // HW, non-zero = object
  std::array<float, 9> k{};        // 3x3 row-major intrinsics
};

// Synthesize an RGB-D frame with a centered square object at a fixed depth.
// frame_index applies a small horizontal shift so tracking has motion to follow.
Frame fillFrame(int width, int height, int frame_index) {
  Frame f;
  f.width = width;
  f.height = height;
  f.rgb.assign(static_cast<size_t>(width) * height * 3, 96);
  f.depth.assign(static_cast<size_t>(width) * height, 0.0f);
  f.mask.assign(static_cast<size_t>(width) * height, 0);

  f.k = {static_cast<float>(width), 0.0f, width * 0.5f,
         0.0f, static_cast<float>(width), height * 0.5f,
         0.0f, 0.0f, 1.0f};

  const int box = std::min(width, height) / 3;
  const int shift = frame_index * 2;
  const int x0 = (width - box) / 2 + shift;
  const int y0 = (height - box) / 2;
  for (int y = y0; y < y0 + box && y < height; ++y) {
    for (int x = x0; x < x0 + box && x < width; ++x) {
      const int idx = y * width + x;
      f.rgb[idx * 3 + 0] = 180;
      f.rgb[idx * 3 + 1] = 130;
      f.rgb[idx * 3 + 2] = 90;
      f.depth[idx] = 0.7f;
      f.mask[idx] = 255;
    }
  }
  return f;
}

fp_image_view_t viewOfSynth(const Frame& f, bool with_mask) {
  fp_image_view_t view{};
  view.width = f.width;
  view.height = f.height;
  view.rgb_u8 = f.rgb.data();
  view.depth_m = f.depth.data();
  view.mask_u8 = with_mask ? f.mask.data() : nullptr;
  view.k_row_major = f.k.data();
  return view;
}

// ---- Pose printing ---------------------------------------------------------
void printPose(int frame_num, const char* phase, float score,
               const float pose[16]) {
  // Print the 3x4 rotation+translation block of the 4x4 row-major SE(3) matrix.
  std::printf(
      "frame %d/%d  %-8s  score=%8.3f  pose: "
      "[%.4f %.4f %.4f %.4f / %.4f %.4f %.4f %.4f / %.4f %.4f %.4f %.4f]\n",
      frame_num, NUMBER_OF_FRAMES, phase, score,
      pose[0], pose[1], pose[2],  pose[3],
      pose[4], pose[5], pose[6],  pose[7],
      pose[8], pose[9], pose[10], pose[11]);
}

// Runs one synthetic frame (generated in code) through register or track.
int processSyntheticFrame(fp_handle_t* handle, int frame_index,
                          bool is_register_frame, fp_pose_result_t& result,
                          std::array<char, 512>& err) {
  Frame frame = fillFrame(kSyntheticWidth, kSyntheticHeight, frame_index);
  fp_image_view_t view = viewOfSynth(frame, is_register_frame);
  if (is_register_frame) {
    return fp_register_frame(handle, &view, -1, -1, &result, err.data(), err.size());
  }
  return fp_track_frame(handle, &view, -1, &result, err.data(), err.size());
}

// Loads one BOP YCB-V frame (scene 000048) from PNGs and runs register or track.
int processBopFrame(fp_handle_t* handle, int frame_index, bool is_register_frame,
                    const std::vector<int>& bop_frame_ids,
                    fp_pose_result_t& result, std::array<char, 512>& err) {
  const std::string im_id_str = std::format("{:06d}", bop_frame_ids[frame_index]);
  const std::string rgb_path = kBopDir + "/rgb/" + im_id_str + ".png";
  const std::string depth_path = kBopDir + "/depth/" + im_id_str + ".png";

  int w = 0;
  int h = 0;
  int dw = 0;
  int dh = 0;
  std::vector<uint8_t> rgb = sample::loadRgb(rgb_path, w, h);
  std::vector<float> depth_m = sample::loadDepthMeters(depth_path, DEPTH_SCALE, dw, dh);

  fp_image_view_t view{};
  view.width = w;
  view.height = h;
  view.rgb_u8 = rgb.data();
  view.depth_m = depth_m.data();
  view.k_row_major = CAMERA_K;

  if (is_register_frame) {
    // gt_id=0: obj_id=1 is the first (and only) object in scene 000048.
    const std::string reg_id_str = std::format("{:06d}", bop_frame_ids[0]);
    const std::string mask_path = kBopDir + "/mask_visib/" + reg_id_str + "_000000.png";
    int mw = 0;
    int mh = 0;
    std::vector<uint8_t> mask = sample::loadMask(mask_path, mw, mh);
    view.mask_u8 = mask.data();
    return fp_register_frame(handle, &view, -1, -1, &result, err.data(), err.size());
  }
  return fp_track_frame(handle, &view, -1, &result, err.data(), err.size());
}

}  // namespace

int main(int argc, char** argv) {
  // ---- Arg parsing -----------------------------------------------------------
  bool use_synthesize = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--use_synthesize") use_synthesize = true;
  }

  std::printf("FoundationPose build: %s\n", fp_build_info());
  std::printf("mode: %s\n", use_synthesize ? "synthetic" : "BOP YCB-V scene 000048");

  // ---- Validate data availability (non-synthetic only) ----------------------
  if (!use_synthesize && !std::filesystem::exists(kBopDir)) {
    std::fprintf(stderr,
                 "error: BOP data not found at %s\n"
                 "Run:  scripts/download_bop_ycbv.sh  (downloads to data/BOP_datasets/)\n",
                 kBopDir.c_str());
    return 1;
  }

  std::array<char, 512> err{};

  // ---- Config ----------------------------------------------------------------
  fp_config_t cfg{};
  fp_default_config(&cfg);
  cfg.max_image_width = kSyntheticWidth;    // 640 covers both modes
  cfg.max_image_height = kSyntheticHeight;  // 480 covers both modes
  cfg.n_hypotheses = nHypotheses();
  std::printf("config: 1 object x %d frames, n_hypotheses=%d\n",
              NUMBER_OF_FRAMES, cfg.n_hypotheses);

  // ---- Create estimator ------------------------------------------------------
  fp_create_options_t opt{};
  opt.cad_path           = kMeshPath.c_str();
  opt.device_id          = 0;
  opt.mesh_unit_scale    = MESH_UNIT_SCALE;
  opt.refine_model_path  = kRefineModel.c_str();
  opt.score_model_path   = kScoreModel.c_str();
  opt.engine_cache_dir   = kEngineCache.c_str();

  fp_handle_t* handle = fp_create(&opt, &cfg, err.data(), err.size());
  if (handle == nullptr) {
    std::cerr << "fp_create failed: " << err.data() << '\n';
    return 1;
  }
  std::printf("created estimator (%s)\n", kMeshPath.c_str());

  if (fp_prepare(handle, cfg.n_hypotheses, err.data(), err.size()) != 0) {
    std::cerr << "fp_prepare failed: " << err.data() << '\n';
    fp_destroy(handle);
    return 1;
  }

  // ---- Collect BOP frame IDs (real-data mode only) ---------------------------
  std::vector<int> bop_frame_ids;
  if (!use_synthesize) {
    try {
      bop_frame_ids = collectFrameIds(kBopDir + "/rgb", NUMBER_OF_FRAMES);
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "error: %s\n", ex.what());
      fp_destroy(handle);
      return 1;
    }
  }

  // ---- Frame loop ------------------------------------------------------------
  bool fatal = false;
  for (int f = 0; f < NUMBER_OF_FRAMES && !fatal; ++f) {
    const bool is_register_frame = (f == 0);
    const char* phase = is_register_frame ? "register" : "track";

    fp_pose_result_t result{};
    const int status =
        use_synthesize
            ? processSyntheticFrame(handle, f, is_register_frame, result, err)
            : processBopFrame(handle, f, is_register_frame, bop_frame_ids, result, err);

    if (status == 0) {
      printPose(f + 1, phase, result.score, result.pose_row_major);
    } else {
      std::fprintf(stderr, "frame %d/%d  %-8s  FAILED: %s\n",
                   f + 1, NUMBER_OF_FRAMES, phase, err.data());
      if (is_register_frame) fatal = true;  // no seed pose -> cannot track
    }
  }

  fp_destroy(handle);
  std::printf("\ndone.\n");
  return fatal ? 1 : 0;
}
