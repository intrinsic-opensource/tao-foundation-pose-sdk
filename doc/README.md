# API Documentation (Doxygen)

The C ABI header is fully Doxygen-annotated with a **Getting Started** main page
(call sequence, minimal example, I/O conventions, error handling, threading notes).

## Generate

```bash
doxygen doc/Doxyfile        # writes doc/api/html/index.html

# Or in container (only Docker required on host):
./run_dev.sh run --rm docs
```

Open `doc/api/html/index.html`. The generated `doc/api/` directory is git-ignored.

For the detailed **register and track data-flow diagrams**, **C ABI / Python integration
workflows** (including multi-object `fp_group_*`), and how to plug in a **custom renderer
or inference backend**, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
