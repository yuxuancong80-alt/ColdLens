param(
    [string]$ArchivePath = "D:\ColdLens\data\raw\microlens50k\MicroLens-50k_covers.zip",
    [string]$OutputPath = "D:\ColdLens\data\processed\cover_dhash.csv"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression.FileSystem

$outputDirectory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
$writer = [System.IO.StreamWriter]::new($OutputPath, $false, [System.Text.UTF8Encoding]::new($false))

try {
    $writer.WriteLine("item,dhash")
    foreach ($entry in $archive.Entries) {
        if ($entry.FullName.EndsWith("/")) { continue }
        $item = [System.IO.Path]::GetFileNameWithoutExtension($entry.Name)
        $stream = $entry.Open()
        $image = $null
        $bitmap = $null
        try {
            $image = [System.Drawing.Image]::FromStream($stream, $true, $true)
            $bitmap = [System.Drawing.Bitmap]::new($image, 9, 8)
            $hex = [System.Text.StringBuilder]::new(16)
            for ($y = 0; $y -lt 8; $y++) {
                $value = 0
                for ($x = 0; $x -lt 8; $x++) {
                    $left = $bitmap.GetPixel($x, $y)
                    $right = $bitmap.GetPixel($x + 1, $y)
                    $leftLuma = 299 * $left.R + 587 * $left.G + 114 * $left.B
                    $rightLuma = 299 * $right.R + 587 * $right.G + 114 * $right.B
                    if ($leftLuma -gt $rightLuma) {
                        $value = $value -bor (1 -shl (7 - $x))
                    }
                }
                [void]$hex.Append($value.ToString("X2"))
            }
            $writer.WriteLine("$item,$hex")
        }
        finally {
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
