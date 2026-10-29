param(
    [ValidateSet('Inventory','Clean')][string]$Phase = 'Inventory',
    [string]$Plan = 'doc/历史实验归档/清理计划.json',
    [string]$Category = '',
    [string]$OnlyPath = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$planPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $Plan))
$spec = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($spec.workspace -ne $projectRoot -or $spec.remote_cleanup -ne $false) { throw 'Unexpected workspace or remote scope' }

function Resolve-Target([string]$relative) {
    if ($relative -notmatch '^(doc|tmp)/[^/]+$' -or $relative.Contains('..')) { throw "Invalid explicit target: $relative" }
    $absolute = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $absolute.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside workspace' }
    foreach ($protected in $spec.protected_directories) {
        $boundary = [IO.Path]::GetFullPath((Join-Path $projectRoot $protected))
        if ($absolute -eq $boundary -or $absolute.StartsWith($boundary + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw "Protected target: $relative" }
    }
    if (Test-Path -LiteralPath $absolute) {
        $item = Get-Item -LiteralPath $absolute -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse target refused: $relative" }
        if ($item.PSIsContainer -and @(Get-ChildItem -LiteralPath $absolute -Recurse -Force | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count) { throw "Nested reparse point refused: $relative" }
    }
    return $absolute
}

function Compact-Value($value, [int]$depth=0) {
    if ($null -eq $value) { return $null }
    if ($depth -ge 14) { return '[deeper diagnostic detail omitted]' }
    if ($value -is [string]) { if ($value.Length -gt 4000) { return $value.Substring(0,4000)+' [long diagnostic string omitted]' }; return $value }
    if ($value -is [ValueType]) { return $value }
    if ($value -is [System.Collections.IList]) {
        if ($value.Count -gt 64) { return [ordered]@{omitted_items=$value.Count;reason='Per-example/long diagnostic array not retained'} }
        $list = [Collections.Generic.List[object]]::new()
        foreach ($child in $value) { $list.Add((Compact-Value $child ($depth+1))) }
        return ,$list.ToArray()
    }
    $properties = @($value.PSObject.Properties)
    if ($properties.Count -gt 80) { return [ordered]@{omitted_object_entries=$properties.Count;reason='Large per-example/code manifest dictionary not retained'} }
    $result = [ordered]@{}
    foreach ($entry in $value.PSObject.Properties) {
        if ($entry.Name -match '(?i)(predictions$|outcomes$|^per_query$|query_ids$|^source_hashes$|^files$|^history$|^per_image$|^image_paths$|^regions$|^cases$)') {
            $result[$entry.Name] = [ordered]@{omitted=$true;reason='Raw per-example/provenance detail; this is a curated summary, not an original seal'}
        } else { $result[$entry.Name] = Compact-Value $entry.Value ($depth+1) }
    }
    return $result
}

function Protected-Fingerprint {
    $paths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($prefix in @('src','scripts','tests','config','.agents') + @($spec.protected_directories)) {
        $folder = Join-Path $projectRoot $prefix
        if (Test-Path -LiteralPath $folder -PathType Container) {
            foreach ($file in Get-ChildItem -LiteralPath $folder -Recurse -File -Force) {
                if ($file.FullName -notmatch '[\\/]__pycache__[\\/]') { [void]$paths.Add($file.FullName) }
            }
        }
    }
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $projectRoot 'doc') -File) { [void]$paths.Add($file.FullName) }
    $hashes = [ordered]@{}
    foreach ($file in ($paths | Sort-Object)) { $hashes[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash }
    return $hashes
}

$targets = foreach ($entry in $spec.targets) {
    $absolute = Resolve-Target $entry.path
    if (-not (Test-Path -LiteralPath $absolute)) { continue }
    $item = Get-Item -LiteralPath $absolute
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $absolute -Recurse -File -Force) } else { @($item) }
    [pscustomobject]@{Entry=$entry;Absolute=$absolute;Item=$item;Files=$files;Bytes=[long](($files | Measure-Object Length -Sum).Sum)}
}

if ($Phase -eq 'Inventory') {
    $records = foreach ($target in $targets) {
        if ($Category -and $target.Entry.category -ne $Category) { continue }
        if ($OnlyPath -and $target.Entry.path -ne $OnlyPath) { continue }
        $summaries = [Collections.Generic.List[object]]::new()
        $notes = [Collections.Generic.List[object]]::new()
        foreach ($file in $target.Files) {
            $relative = [IO.Path]::GetRelativePath($projectRoot,$file.FullName).Replace('\','/')
            if ($relative -match '/source/' -or $file.Length -gt 8MB) { continue }
            if ($file.Name -match '(?i)^(summary|.*_summary|report|decision|statistics|metrics|results)\.json$') {
                try {
                    $data = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json -Depth 100
                    $summaries.Add([ordered]@{source=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant();data=(Compact-Value $data)})
                } catch { throw "Cannot preserve summary $relative : $_" }
            } elseif ($file.Name -match '(?i)^(summary|paired_comparisons|verdict).*\.(csv|txt)$' -and $file.Length -le 128KB) {
                $summaries.Add([ordered]@{source=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant();text=(Get-Content -LiteralPath $file.FullName -Raw)})
            } elseif ($file.Extension -eq '.md' -and $file.Length -le 128KB) {
                $notes.Add([ordered]@{source=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant();text=(Get-Content -LiteralPath $file.FullName -Raw)})
            }
        }
        [ordered]@{path=$target.Entry.path;category=$target.Entry.category;files=$target.Files.Count;bytes=$target.Bytes;summaries=$summaries.ToArray();notes=$notes.ToArray()}
    }
    [ordered]@{schema='local_experiment_cleanup_inventory_v1';workspace=$projectRoot;records=@($records)} | ConvertTo-Json -Depth 100 -Compress
    exit
}

if ($spec.reviewed -ne $true) { throw 'Review/curated archives required before cleanup' }
$inventory = Get-Content -LiteralPath (Join-Path $projectRoot 'doc/历史实验归档/清理清单.json') -Raw | ConvertFrom-Json
foreach ($archive in $spec.curated_archives) {
    $saved = Get-Content -LiteralPath (Join-Path $projectRoot $archive.path) -Raw | ConvertFrom-Json
    if ($saved.scope -ne 'curated_metrics_not_original_evidence' -or $saved.experiments.Count -ne $archive.experiments) { throw "Invalid retained archive: $($archive.path)" }
    foreach ($experiment in $saved.experiments) {
        foreach ($record in @($experiment.summaries) + @($experiment.notes)) {
            $original = [IO.Path]::GetFullPath((Join-Path $projectRoot $record.source))
            if (-not $original.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive source outside workspace' }
            if ((Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) { throw "Summary changed after archival: $($record.source)" }
        }
    }
}
$tracked = @(git -C $projectRoot -c core.quotePath=false ls-files)
foreach ($target in $targets) {
    $record = @($inventory.targets | Where-Object {$_.path -eq $target.Entry.path})
    if ($record.Count -ne 1 -or $record[0].files -ne $target.Files.Count -or $record[0].bytes -ne $target.Bytes) { throw "Target changed since inventory: $($target.Entry.path)" }
    if (@($tracked | Where-Object {$_ -eq $target.Entry.path -or $_.StartsWith($target.Entry.path+'/')}).Count) { throw "Tracked target requires separate review: $($target.Entry.path)" }
}
$before = Protected-Fingerprint
Add-Type -AssemblyName Microsoft.VisualBasic
$removed = [Collections.Generic.List[object]]::new()
foreach ($target in $targets) {
    if ($target.Item.PSIsContainer) {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($target.Absolute,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,
            [Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
    } else {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($target.Absolute,
            [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
            [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,
            [Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
    }
    $removed.Add([ordered]@{path=$target.Entry.path;files=$target.Files.Count;bytes=$target.Bytes})
}
$after = Protected-Fingerprint
$changed = @($before.Keys | Where-Object {-not $after.Contains($_) -or $after[$_] -ne $before[$_]})
$added = @($after.Keys | Where-Object {-not $before.Contains($_)})
[ordered]@{completed_at=(Get-Date -Format o);mode='Windows_Recycle_Bin';remote_touched=$false;
    removed=$removed.ToArray();protected_files=$before.Count;protected_unchanged=($changed.Count -eq 0 -and $added.Count -eq 0);
    changed=$changed;added=$added;code_tests_run=$false;git_history_rewritten=$false} | ConvertTo-Json -Depth 8 -Compress
if ($changed.Count -or $added.Count) { throw 'Protected files changed during cleanup; inspect receipt and restore from Recycle Bin' }
