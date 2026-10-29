$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$plan=Get-Content -LiteralPath (Join-Path $root 'doc/历史实验归档/二次整理计划.json') -Raw | ConvertFrom-Json
$protected=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach($entry in $plan.keep_root){if($entry -notin @('doc/README.md','doc/项目实验总览.md','doc/EXPERIMENT_ARTIFACT_POLICY.md')){[void]$protected.Add($entry)}}
$f1=Get-Content -LiteralPath (Join-Path $root 'doc/创新点1/archive_manifest.json') -Raw | ConvertFrom-Json
foreach($entry in $f1.files.PSObject.Properties){[void]$protected.Add('doc/创新点1/'+$entry.Name)}
$changes=[Collections.Generic.List[object]]::new()
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'doc') -Filter '*.md' -Recurse -File) {
    $relative=[IO.Path]::GetRelativePath($root,$file.FullName).Replace('\','/')
    if($protected.Contains($relative) -or $relative.StartsWith('doc/创新点2/交接包/')){continue}
    $moved=@($plan.moves | Where-Object {$_.destination -eq $relative})
    $original=if($moved.Count -eq 1){$moved[0].source}else{$relative}
    $oldParent=[IO.Path]::GetDirectoryName((Join-Path $root $original))
    $before=[IO.File]::ReadAllText($file.FullName)
    $after=[regex]::Replace($before,'(?<prefix>!?\[[^\]\r\n]*\]\()(?<target><[^>]+>|[^\s)]+)(?<suffix>\))',{
        param($match)
        $raw=$match.Groups['target'].Value
        $target=$raw.Trim('<','>')
        if($target -match '^(https?://|mailto:|#|codex:)' -or $target.Contains('://')){return $match.Value}
        $anchor='';if($target.Contains('#')){$parts=$target.Split('#',2);$target=$parts[0];$anchor='#'+$parts[1]}
        if(-not $target){return $match.Value}
        try{$absolute=[IO.Path]::GetFullPath((Join-Path $oldParent $target))}catch{return $match.Value}
        if(-not $absolute.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)){return $match.Value}
        $dest=[IO.Path]::GetRelativePath($root,$absolute).Replace('\','/')
        foreach($mapping in $plan.moves){
            if($dest -eq $mapping.source){$dest=$mapping.destination;break}
            if($dest.StartsWith($mapping.source+'/')){$dest=$mapping.destination+$dest.Substring($mapping.source.Length);break}
        }
        foreach($mapping in $plan.report_archives){if($dest -eq $mapping.source -or $dest.StartsWith($mapping.source+'/')){$dest=$mapping.destination;$anchor='';break}}
        $new=[IO.Path]::GetRelativePath($file.DirectoryName,(Join-Path $root $dest)).Replace('\','/')+$anchor
        if($new -match '\s'){$new='<'+$new+'>'}
        return $match.Groups['prefix'].Value+$new+$match.Groups['suffix'].Value
    })
    if($after -ne $before){$changes.Add([ordered]@{path=$relative;before=$before.Replace("`r`n","`n");after=$after.Replace("`r`n","`n")})}
}
$changes.ToArray() | ConvertTo-Json -Depth 4 -Compress
