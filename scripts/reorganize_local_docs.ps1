param([ValidateSet('Archive','Organize','PruneTransfers')][string]$Phase='Archive')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$plan=Get-Content -LiteralPath (Join-Path $root 'doc/历史实验归档/二次整理计划.json') -Raw | ConvertFrom-Json
if ($root -ne $plan.workspace) {throw 'Wrong workspace'}
function Safe([string]$rel) {
    if ([IO.Path]::IsPathRooted($rel) -or $rel.Split('/').Contains('..')) {throw 'Invalid relative target'}
    $p=[IO.Path]::GetFullPath((Join-Path $root $rel))
    if (-not $p.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Outside workspace'}
    if ($rel -match '^(\.git|src|scripts|config|tests|data|logs|\.cache)(/|$)') {throw 'Code/data/cache/history outside this script scope'}
    if (Test-Path -LiteralPath $p) {
        if ((Get-Item -LiteralPath $p -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Reparse point refused'}
        if (Test-Path -LiteralPath $p -PathType Container) {
            if (@(Get-ChildItem -LiteralPath $p -Recurse -Force | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count) {throw 'Nested reparse point refused'}
        }
    }
    return $p
}
function Hash-Stream($stream) { $sha=[Security.Cryptography.SHA256]::Create();try{return [Convert]::ToHexString($sha.ComputeHash($stream)).ToLowerInvariant()}finally{$sha.Dispose()} }
function File-Hash([string]$p) { return (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() }
function Zip-Snapshot([string]$destination,$files,[string]$entryBase) {
    $target=Safe $destination
    if (Test-Path -LiteralPath $target) {throw "Refusing archive overwrite: $destination"}
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
    $records=@(foreach ($file in $files) { [ordered]@{path=[IO.Path]::GetRelativePath($entryBase,$file.FullName).Replace('\','/');sha256=(File-Hash $file.FullName);bytes=$file.Length} })
    $zip=[IO.Compression.ZipFile]::Open($target,[IO.Compression.ZipArchiveMode]::Create)
    try { foreach ($file in $files) {
        $entry=[IO.Path]::GetRelativePath($entryBase,$file.FullName).Replace('\','/')
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$file.FullName,$entry,[IO.Compression.CompressionLevel]::Optimal)
    }} finally {$zip.Dispose()}
    $zip=[IO.Compression.ZipFile]::OpenRead($target)
    try {
        if ($zip.Entries.Count -ne $records.Count) {throw 'Archive count differs'}
        foreach ($record in $records) {
            $entry=$zip.GetEntry($record.path)
            if ($null -eq $entry -or $entry.Length -ne $record.bytes) {throw 'Archive entry differs'}
            $stream=$entry.Open();try{if ((Hash-Stream $stream) -ne $record.sha256) {throw 'Archive content differs'}}finally{$stream.Dispose()}
        }
    } finally {$zip.Dispose()}
    return [ordered]@{archive=$destination;sha256=(File-Hash $target);files=$records;bytes=(Get-Item -LiteralPath $target).Length}
}

if ($Phase -eq 'Archive') {
    $archives=[Collections.Generic.List[object]]::new()
    $mds=@(Get-ChildItem -LiteralPath (Join-Path $root 'doc') -File -Filter '*.md')
    $archives.Add((Zip-Snapshot 'doc/历史实验归档/20261029_整理前根目录说明.zip' $mds $root))
    foreach ($mapping in $plan.report_archives) {
        $source=Safe $mapping.source
        $files=@(Get-ChildItem -LiteralPath $source -File -Recurse -Force)
        $snapshot=Zip-Snapshot $mapping.destination $files $source
        $snapshot['source']=$mapping.source
        $archives.Add($snapshot)
    }
    [ordered]@{phase='archive_verified';archives=$archives.ToArray();deletion_started=$false;remote_touched=$false} | ConvertTo-Json -Depth 8 -Compress
    exit
}

$receipt=Get-Content -LiteralPath (Join-Path $root 'doc/历史实验归档/二次整理压缩核验.json') -Raw | ConvertFrom-Json
if ($Phase -eq 'PruneTransfers') {
    $duplicates=[Collections.Generic.List[object]]::new()
    foreach ($snapshot in $receipt.archives) {
        if (-not $snapshot.source) {continue}
        $stem=($snapshot.source.Split('/')[-1] -replace '_server$','')
        $relative='tmp/'+$stem+'_reports.tar.gz';$transfer=Safe $relative
        if (-not (Test-Path -LiteralPath $transfer -PathType Leaf)) {continue}
        $zipPath=Safe $snapshot.archive
        if ((File-Hash $zipPath) -ne $snapshot.sha256) {throw 'ZIP changed'}
        $zip=[IO.Compression.ZipFile]::OpenRead($zipPath)
        $stream=[IO.File]::OpenRead($transfer)
        $gzip=[IO.Compression.GZipStream]::new($stream,[IO.Compression.CompressionMode]::Decompress)
        $tar=[System.Formats.Tar.TarReader]::new($gzip)
        $checked=0
        try {
            while ($entry=$tar.GetNextEntry()) {
                if ($null -eq $entry.DataStream) {continue}
                $name=$entry.Name.Replace('\','/');while($name.StartsWith('./')){$name=$name.Substring(2)}
                if ($name.Split('/').Contains('..')) {throw 'Unsafe tar entry'}
                $other=$zip.GetEntry($name)
                if ($null -eq $other -or $other.Length -ne $entry.Length) {throw "Transfer not redundant: $relative : $name"}
                $member=$other.Open()
                try {if ((Hash-Stream $entry.DataStream) -ne (Hash-Stream $member)) {throw 'Transfer content differs'}} finally {$member.Dispose()}
                $checked++
            }
        } finally {$tar.Dispose();$gzip.Dispose();$stream.Dispose();$zip.Dispose()}
        if ($checked -lt 1) {throw 'Empty transfer verification'}
        $duplicates.Add([ordered]@{path=$relative;sha256=(File-Hash $transfer);bytes=(Get-Item -LiteralPath $transfer).Length;entries_verified=$checked;retained_archive=$snapshot.archive})
    }
    foreach ($item in $duplicates) {Remove-Item -LiteralPath (Safe $item.path) -Force}
    [ordered]@{phase='duplicate_transfers_removed';removed=$duplicates.ToArray();retained_content_verified=$true} | ConvertTo-Json -Depth 5 -Compress
    exit
}
foreach ($snapshot in $receipt.archives) {
    if ((File-Hash (Safe $snapshot.archive)) -ne $snapshot.sha256) {throw 'Verified archive changed'}
    if ($snapshot.source) {
        $source=Safe $snapshot.source
        $actual=@(Get-ChildItem -LiteralPath $source -File -Recurse -Force)
        if ($actual.Count -ne $snapshot.files.Count) {throw 'Source changed after verification'}
        foreach ($file in $snapshot.files) {
            $path=Join-Path $source $file.path
            if ((File-Hash $path) -ne $file.sha256) {throw 'Source changed after verification'}
        }
    }
}
# All deletions below are expanded local report copies with byte-verified ZIPs.
$deleted=[Collections.Generic.List[object]]::new()
foreach ($mapping in $plan.report_archives) {
    $source=Safe $mapping.source
    if ($mapping.source -notmatch '^doc/(scale_|cross_resolution_)' -or $mapping.destination -notlike 'doc/创新点2/证据包/*') {throw 'Not an approved report-copy target'}
    Remove-Item -LiteralPath $source -Recurse -Force
    $deleted.Add($mapping.source)
}
$moved=[Collections.Generic.List[object]]::new()
foreach ($mapping in $plan.moves) {
    $source=Safe $mapping.source;$destination=Safe $mapping.destination
    if (Test-Path -LiteralPath $destination) {throw "Destination already exists: $destination"}
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))
    $before=if (Test-Path -LiteralPath $source -PathType Leaf) {File-Hash $source} else {''}
    Move-Item -LiteralPath $source -Destination $destination
    if ($before -and (File-Hash $destination) -ne $before) {throw 'Move content differs'}
    $moved.Add([ordered]@{source=$mapping.source;destination=$mapping.destination;sha256=$before})
}
[ordered]@{phase='organized';deleted_expanded_copies=$deleted.ToArray();moved=$moved.ToArray();remote_touched=$false;source_code_changed=$false;git_history_rewritten=$false} | ConvertTo-Json -Depth 8 -Compress
