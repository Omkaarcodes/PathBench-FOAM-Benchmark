# run_all_benchmarks.ps1
# Runs every non-NN algorithm on every non-3D map (one command per combination),
# then compiles all results into summary CSVs.
#
# Usage (from repo root):
#   .\run_all_benchmarks.ps1
#   .\run_all_benchmarks.ps1 -TrialsPerMap 5 -OutputDir "data/my_results"

param(
    [int]$TrialsPerMap = 10,
    [int]$Seed = 42,
    [string]$OutputDir = "data/research_results",
    [int]$TimeoutSeconds = 180,  # per (map, algorithm) combo
    [int]$BackupEveryCombos = 10
)

$ErrorActionPreference = "Continue"

$PYTHON = "c:/Users/Omkaar Sampigeadi/Documents/PathBench-master/.venv/Scripts/python.exe"
$SCRIPT = "src/run_research_benchmarks.py"
$RAW_CSV = Join-Path $OutputDir "raw_runs.csv"
$BACKUP_DIR = Join-Path $OutputDir "backups"

function Quote-WinArg {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    # Escape quotes and preserve trailing backslashes for Windows command-line parsing.
    $escaped = $Value -replace '(\\*)"', '$1$1\\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Save-RawRunsBackup {
    param(
        [string]$Reason
    )

    if (-not (Test-Path $RAW_CSV)) {
        return
    }

    New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeReason = ($Reason -replace '[^A-Za-z0-9_-]', '_')
    $dst = Join-Path $BACKUP_DIR ("raw_runs_{0}_{1}.csv" -f $safeReason, $stamp)
    Copy-Item $RAW_CSV $dst -Force
    Write-Host "  [BACKUP] $dst" -ForegroundColor DarkCyan
}

# Non-NN algorithms only
$ALGORITHMS = @(
    'A*'
    'SPRM'
    'RT'
    'RRT'
    'RRT*'
    'RRT-Connect'
    'Wave-front'
    'Dijkstra'
    'Potential Field'
    'Probabilistic Foam (GBPF)'
    'Probabilistic Foam (PFM)'
    'Probabilistic Foam (RBPF)'
    'Probabilistic Foam (HPF)'
)

# All built-in non-3D maps
$MAPS = @(
    'Uniform Random Fill'
    'Block'
    'House'
    'Long Wall'
    'Labyrinth'
    'vin test 8x8'
    'vin test 8x8 -2'
    'vin test 8x8 -3'
    'vin test 16x16 -1'
    'vin test 16x16 -2'
    'vin test 28x28 -1'
    'Small Obstacle'
    'Occupancy Grid 2D'
    'SLAM Map 1'
    'SLAM Map 1 (compressed)'
    'SLAM Map 2'
    'SLAM Map 3'
    'Urban City Grid'
    'Urban Downtown'
    'Urban Parking Lot'
    'Urban City Park'
    'Urban T-Intersection'
    'Urban Roundabout'
    'Urban Construction Zone'
    'Urban Industrial Zone'
    'Urban Narrow Alleys'
    'Urban Suburban'
    'Urban Highway Corridor'
    'Urban Crossroads'
)

$Total   = $ALGORITHMS.Count * $MAPS.Count
$Done    = 0
$Failed  = 0

Write-Host ""
Write-Host "PathBench Research Benchmark Runner" -ForegroundColor Cyan
Write-Host "  Algorithms : $($ALGORITHMS.Count)" -ForegroundColor Cyan
Write-Host "  Maps       : $($MAPS.Count)"       -ForegroundColor Cyan
Write-Host "  Combos     : $Total"               -ForegroundColor Cyan
Write-Host "  Trials/map : $TrialsPerMap"        -ForegroundColor Cyan
Write-Host "  Timeout    : ${TimeoutSeconds}s per combo" -ForegroundColor Cyan
Write-Host "  Backup     : every $BackupEveryCombos combos" -ForegroundColor Cyan
Write-Host "  Output dir : $OutputDir"           -ForegroundColor Cyan
Write-Host ""

foreach ($Map in $MAPS) {
    foreach ($Algo in $ALGORITHMS) {
        $Done++
        $Pct = [math]::Round(($Done / $Total) * 100, 1)
        Write-Host "[$Done/$Total  $Pct%]  map=$Map  algo=$Algo" -ForegroundColor Yellow

        $CmdArgs = @(
            $SCRIPT,
            '--map', $Map,
            '--algorithm', $Algo,
            '--trials-per-map', "$TrialsPerMap",
            '--seed', "$Seed",
            '--output-dir', $OutputDir
        )
        $ArgLine = ($CmdArgs | ForEach-Object { Quote-WinArg $_ }) -join ' '

        $proc = Start-Process -FilePath $PYTHON -ArgumentList $ArgLine -NoNewWindow -PassThru
        $finished = $proc.WaitForExit($TimeoutSeconds * 1000)
        if (-not $finished) {
            Write-Host "  [TIMEOUT] Killing after ${TimeoutSeconds}s  map=$Map  algo=$Algo" -ForegroundColor Magenta
            $proc | Stop-Process -Force -ErrorAction SilentlyContinue
            $Failed++
            Save-RawRunsBackup -Reason "timeout"
        } else {
            $exitCode = 0
            try {
                $exitCode = [int]$proc.ExitCode
            } catch {
                $exitCode = 1
            }
            if ($exitCode -ne 0) {
                Write-Host "  [WARN] Non-zero exit ($exitCode) for map=$Map  algo=$Algo" -ForegroundColor Red
                $Failed++
                Save-RawRunsBackup -Reason "nonzero_exit"
            }
        }

        if (($Done % $BackupEveryCombos) -eq 0) {
            Save-RawRunsBackup -Reason "checkpoint"
        }
    }
}

Write-Host ""
Write-Host "All combinations finished.  Failed: $Failed / $Total" -ForegroundColor Cyan
Write-Host "Compiling summary CSVs..." -ForegroundColor Cyan

& $PYTHON $SCRIPT --compile --output-dir $OutputDir
Save-RawRunsBackup -Reason "final"

Write-Host ""
Write-Host "Done. Results saved to: $OutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "Output files:" -ForegroundColor Green
Write-Host "  raw_runs.csv                  (one row per trial)"
Write-Host "  per_map_algorithm_summary.csv (mean/std per map x algorithm)"
Write-Host "  foam_vs_nonfoam_urban.csv     (FOAM vs non-FOAM on urban maps)"
Write-Host "  foam_vs_nonfoam_nonurban.csv  (FOAM vs non-FOAM on non-urban maps)"
Write-Host "  learning_vs_classical_summary.csv"
