set HF_HOME=./src_models/omnicoder-9b-q4_k_m/cache

llama-server ^
  -m "C:\Users\mfour\.lmstudio\models\Tesslate\OmniCoder-9B-GGUF\omnicoder-9b-q4_k_m.gguf" ^
  --ctx-size 32768 ^
  --batch-size 16358 ^
  --ubatch-size 2048 ^
  --jinja ^
  --cache-prompt ^
  --context-shift ^
  --cont-batching ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -fa on ^
  -ngl 99 ^
  --port 8002