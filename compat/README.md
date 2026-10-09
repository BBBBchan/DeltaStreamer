# Inference compatibility files

This file retains the current runtime modifications to ModelScope
ms-swift 4.2.0. The copyright header and [Apache-2.0 license](LICENSE) are preserved.
The complete framework is fetched by `scripts/setup_runtime.py`, not vendored
in the release. The setup script constructs an isolated `.runtime/swift`
package and overlays this file without modifying installed packages.
