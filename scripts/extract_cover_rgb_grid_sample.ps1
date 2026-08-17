param(
    [string]$ArchivePath = "D:\ColdLens\data\raw\microlens50k\MicroLens-50k_covers.zip",
    [string]$OutputPath = "D:\ColdLens\outputs\visual_smoke\rgb_grid_8x8_sample.csv"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression.FileSystem

# Fixed sample: known audit pairs plus a deterministic spread across the ID range.
$sampleIds = @(
    46, 313, 792, 902, 1186, 1792, 1852, 1864,
    2491, 3478, 3658, 3683, 4141, 4237, 5036, 5628,
    6061, 6443, 7157, 7979, 8425, 8557, 8714, 9319,
    10204, 11800, 13064, 13443, 15000, 16683, 17620, 18187
)

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
$writer = [System.IO.StreamWriter]::new($OutputPath, $false, [System.Text.UTF8Encoding]::new($false))

try {
    $header = [System.Collections.Generic.List[string]]::new()
    $header.Add("item")
    for ($y = 0; $y -lt 8; $y++) {
        for ($x = 0; $x -lt 8; $x++) {
            foreach ($channel in @("r", "g", "b")) {
                $header.Add("p${y}_${x}_${channel}")
            }
        }
    }
    $writer.WriteLine(($header -join ","))

    foreach ($item in $sampleIds) {
        $entry = $archive.GetEntry("MicroLens-50k_covers/$item.jpg")
        if (-not $entry) { throw "Missing cover for item $item" }
        $stream = $entry.Open()
        $image = $null
        $bitmap = $null
        $graphics = $null
        try {
            $image = [System.Drawing.Image]::FromStream($stream, $true, $true)
            $bitmap = [System.Drawing.Bitmap]::new(8, 8, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $graphics.DrawImage($image, 0, 0, 8, 8)
            $values = [System.Collections.Generic.List[string]]::new()
            $values.Add($item.ToString())
            for ($y = 0; $y -lt 8; $y++) {
                for ($x = 0; $x -lt 8; $x++) {
                    $pixel = $bitmap.GetPixel($x, $y)
                    $values.Add($pixel.R.ToString())
                    $values.Add($pixel.G.ToString())
                    $values.Add($pixel.B.ToString())
                }
            }
            $writer.WriteLine(($values -join ","))
        }
        finally {
            if ($graphics) { $graphics.Dispose() }
            if ($bitmap) { $bitmap.Dispose() }
            if ($image) { $image.Dispose() }
            $stream.Dispose()
        }
    }
}
finally {
    $writer.Dispose()
    $archive.Dispose()
}

Get-Item -LiteralPath $OutputPath | Select-Object FullName, Length
