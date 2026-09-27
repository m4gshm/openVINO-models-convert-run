param(
    [string]$ModelDir,
    [string]$ConfigFile
)

# Use forward slashes for JSON (OVMS accepts both)
$forwardSlashPath = $ModelDir -replace '\\', '/'

$json = @"
{
    "model_config_list": [
        {
            "config": {
                "base_path": "$forwardSlashPath",
                "model_path": "openvino_model.xml",
                "stateful": true,
                "batch_size": "auto"
            },
            "version": "1",
            "aliases": ["primary"]
        }
    ]
}
"@

# Write without BOM - use ASCII encoding for pure ASCII JSON
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json.TrimEnd())
[System.IO.File]::WriteAllBytes($ConfigFile, $bytes)
