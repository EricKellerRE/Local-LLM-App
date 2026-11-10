# Moving Local Model to another Windows machine

The app supports direct GGUF files using its own standalone llama.cpp process.
LM Studio is not required. Select **GGUF (llama.cpp)** in Settings and enter the
full language-model file path. For split models select `00001-of-...`; keep all
shards in the same folder. Do not select an `mmproj` vision projector. This
backend currently handles text and tools, not image inputs.

The bundled CPU runtime is llama.cpp b12144, downloaded from the official
ggml-org/llama.cpp release. Its source archive SHA-256 is
`569990fcbacbe305aef41490f71df70088474a59d514ae3b1ee65c130ee74555`.
Set `LOCAL_LLAMA_SERVER` to a different standalone Windows build to use CUDA or
Vulkan on the new machine. CPU is the tested configuration. The app launches
the runtime on a private loopback port and closes it during normal shutdown.
Stopping GGUF inference terminates only its owned runtime; the next request
reloads it. A forcibly killed app can leave a child process behind.

## Transfer

Run from this repository with its Python environment:

```
localmodel-env\python.exe scripts\export_portable.py
```

This creates `portable-dist/Local-Model-Windows.zip`: current app code, the
standalone CPU runtime, and sibling Grid-Workshop tracked source files. It
excludes credentials, Python environments, cases, research artifacts, and chats.
An inventory in the ZIP records exclusions. No source files are changed.

To include GGUF files, repeat `--model PATH` for each desired language model;
all shards of a selected split model are included automatically. Models are
large; copying them separately to `Local LLM/models/` is usually easier.
Optionally pass `--include-data` **only after closing the app**, to include saved
chats and activity. Old chats/project references can contain paths to the old
computer; select new case/project paths when resuming them. Do not resume an
old study blindly. The transfer contains no PowerWorld license or installation.

## Destination setup

1. Extract the ZIP, preserving sibling `Local LLM` and `Grid-Workshop` folders.
2. Install Python 3.12 x64, then run `Install Local LLM.cmd`. Internet access is
   needed to install dependencies; Python virtual environments must be rebuilt.
3. Copy your GGUFs to `Local LLM/models/` if not included in the transfer.
4. Run `Local LLM.cmd`, select a GGUF in Settings, and choose a context/output
   budget appropriate to the machine. Start with 16,384 context and 2,048 output.
   Settings save automatically. Restart the app to load the selected model and
   apply runtime changes.
5. Install and license PowerWorld Simulator and register SimAuto separately.
   Enable Grid Workshop using the lightning-bolt button in the intended chat.
6. Run `scripts/grid_analysis_readiness.py` using the installed Python to verify
   the plugin without opening a case. Run `scripts/gguf_smoke.py MODEL_PATH` to
   exercise direct inference and a synthetic tool call.
   `scripts/gguf_workflow_smoke.py` runs the local agent against synthetic CSVs,
   including a rejected call, filtering, aggregation, and deterministic math.
   Add `--case` to create a fresh synthetic PowerWorld case and test the agent's
   discovery/query/aggregation workflow. This requires working SimAuto. These
   tests create their own fixtures; they do not use your study cases.

This is a relocatable source/runtime bundle, not a zero-install executable.
The default CPU backend does not require a GPU. A supported GPU runtime and
drivers are separate destination-specific prerequisites. The 120B GGUF weights
are about 59 GiB before context buffers and OS overhead; more than 64 GB RAM is
advisable for comfortable testing. No speed or result accuracy is guaranteed.
