Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path (Split-Path -Parent $projectDir) "2026-06-17\ai-training-file-preparer\.venv\Scripts\python.exe"
$augmenterPath = Join-Path $projectDir "augment_training_images.py"

function Quote-Argument([string] $Value) {
    if ($Value.Contains('"')) {
        throw "Paths and values containing quotation marks are not supported."
    }
    return '"' + $Value + '"'
}

function Add-Label($Parent, [string] $Text, [int] $X, [int] $Y, [int] $Width = 180) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.Location = New-Object System.Drawing.Point($X, $Y)
    $label.Size = New-Object System.Drawing.Size($Width, 22)
    $Parent.Controls.Add($label)
    return $label
}

function Add-TextBox($Parent, [int] $X, [int] $Y, [int] $Width, [string] $Value = "") {
    $box = New-Object System.Windows.Forms.TextBox
    $box.Location = New-Object System.Drawing.Point($X, $Y)
    $box.Size = New-Object System.Drawing.Size($Width, 24)
    $box.Text = $Value
    $Parent.Controls.Add($box)
    return $box
}

function Add-Button($Parent, [string] $Text, [int] $X, [int] $Y, [int] $Width = 75) {
    $button = New-Object System.Windows.Forms.Button
    $button.Text = $Text
    $button.Location = New-Object System.Drawing.Point($X, $Y)
    $button.Size = New-Object System.Drawing.Size($Width, 27)
    $Parent.Controls.Add($button)
    return $button
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "Morphology-safe TIFF Augmenter"
$form.StartPosition = "CenterScreen"
$form.Size = New-Object System.Drawing.Size(980, 760)
$form.MinimumSize = New-Object System.Drawing.Size(900, 700)
$form.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$left = New-Object System.Windows.Forms.Panel
$left.Dock = "Left"
$left.Width = 460
$left.AutoScroll = $true
$form.Controls.Add($left)

$right = New-Object System.Windows.Forms.Panel
$right.Dock = "Fill"
$right.Padding = New-Object System.Windows.Forms.Padding(10)
$form.Controls.Add($right)

Add-Label $left "Image input" 14 14 200 | Out-Null
$inputBox = Add-TextBox $left 14 38 300 "C:\Users\jds17\OneDrive - UBC\Data_UBC student\Data\2025\Week45 Donor#5\05.AI Fed\Refined\Refined_GCG_Low\Ar10_N19__Ar10_N19_667.tif"
$inputFileButton = Add-Button $left "File" 320 36 55
$inputFolderButton = Add-Button $left "Folder" 380 36 65

Add-Label $left "Mask input (optional)" 14 78 200 | Out-Null
$maskBox = Add-TextBox $left 14 102 300
$maskFileButton = Add-Button $left "File" 320 100 55
$maskFolderButton = Add-Button $left "Folder" 380 100 65
$maskClearButton = Add-Button $left "Clear" 380 130 65

Add-Label $left "Output folder" 14 158 200 | Out-Null
$outputBox = Add-TextBox $left 14 182 360 (Join-Path $projectDir "augmented_files")
$outputButton = Add-Button $left "Browse" 380 180 65

Add-Label $left "Transforms" 14 226 200 | Out-Null
$quarterCheck = New-Object System.Windows.Forms.CheckBox
$quarterCheck.Text = "90 / 180 / 270 degree rotations"
$quarterCheck.Location = New-Object System.Drawing.Point(16, 252)
$quarterCheck.Size = New-Object System.Drawing.Size(260, 24)
$quarterCheck.Checked = $true
$left.Controls.Add($quarterCheck)

$flipCheck = New-Object System.Windows.Forms.CheckBox
$flipCheck.Text = "Horizontal flips"
$flipCheck.Location = New-Object System.Drawing.Point(16, 278)
$flipCheck.Size = New-Object System.Drawing.Size(220, 24)
$flipCheck.Checked = $true
$left.Controls.Add($flipCheck)

$shiftCheck = New-Object System.Windows.Forms.CheckBox
$shiftCheck.Text = "Integer translations"
$shiftCheck.Location = New-Object System.Drawing.Point(16, 306)
$shiftCheck.Size = New-Object System.Drawing.Size(170, 24)
$shiftCheck.Checked = $true
$left.Controls.Add($shiftCheck)
$shiftBox = Add-TextBox $left 195 305 55 "5"
Add-Label $left "pixels (four directions)" 257 308 180 | Out-Null

$smallRotationCheck = New-Object System.Windows.Forms.CheckBox
$smallRotationCheck.Text = "Small rotations +/-"
$smallRotationCheck.Location = New-Object System.Drawing.Point(16, 336)
$smallRotationCheck.Size = New-Object System.Drawing.Size(170, 24)
$smallRotationCheck.Checked = $false
$left.Controls.Add($smallRotationCheck)
$angleBox = Add-TextBox $left 195 335 55 "5"
Add-Label $left "degrees (nearest neighbour)" 257 338 190 | Out-Null

Add-Label $left "Empty-border fill" 16 374 170 | Out-Null
$fillModeBox = New-Object System.Windows.Forms.ComboBox
$fillModeBox.Location = New-Object System.Drawing.Point(195, 371)
$fillModeBox.Size = New-Object System.Drawing.Size(210, 24)
$fillModeBox.DropDownStyle = "DropDownList"
[void] $fillModeBox.Items.Add("Per-plane border median")
[void] $fillModeBox.Items.Add("Constant value")
$fillModeBox.SelectedIndex = 0
$left.Controls.Add($fillModeBox)

Add-Label $left "Constant fill value" 16 406 170 | Out-Null
$fillValueBox = Add-TextBox $left 195 403 55 "0"

$recursiveCheck = New-Object System.Windows.Forms.CheckBox
$recursiveCheck.Text = "Search subfolders"
$recursiveCheck.Location = New-Object System.Drawing.Point(16, 440)
$recursiveCheck.Size = New-Object System.Drawing.Size(180, 24)
$recursiveCheck.Checked = $true
$left.Controls.Add($recursiveCheck)

$overwriteCheck = New-Object System.Windows.Forms.CheckBox
$overwriteCheck.Text = "Overwrite existing files"
$overwriteCheck.Location = New-Object System.Drawing.Point(210, 440)
$overwriteCheck.Size = New-Object System.Drawing.Size(200, 24)
$left.Controls.Add($overwriteCheck)

$warning = Add-Label $left "Recommended for this image: 5-pixel translation, per-plane border median, and small rotation OFF." 16 475 420
$warning.Height = 48
$warning.ForeColor = [System.Drawing.Color]::DarkOrange

$runButton = Add-Button $left "Create augmented images" 16 535 420
$runButton.Height = 38
$runButton.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)

$statusLabel = Add-Label $left "Ready" 16 584 420
$statusLabel.ForeColor = [System.Drawing.Color]::SteelBlue
$statusLabel.Height = 42

$logTitle = Add-Label $right "Log" 4 4 100
$logTitle.Font = New-Object System.Drawing.Font("Segoe UI", 11, [System.Drawing.FontStyle]::Bold)
$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Location = New-Object System.Drawing.Point(4, 32)
$logBox.Size = New-Object System.Drawing.Size(470, 650)
$logBox.Anchor = "Top,Bottom,Left,Right"
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = "Vertical"
$logBox.Font = New-Object System.Drawing.Font("Consolas", 9)
$right.Controls.Add($logBox)

$openFile = {
    param($TargetBox)
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Filter = "TIFF files (*.tif;*.tiff)|*.tif;*.tiff|All files (*.*)|*.*"
    if ($dialog.ShowDialog() -eq "OK") { $TargetBox.Text = $dialog.FileName }
}
$openFolder = {
    param($TargetBox)
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    if ($dialog.ShowDialog() -eq "OK") { $TargetBox.Text = $dialog.SelectedPath }
}

$inputFileButton.Add_Click({ & $openFile $inputBox })
$inputFolderButton.Add_Click({ & $openFolder $inputBox })
$maskFileButton.Add_Click({ & $openFile $maskBox })
$maskFolderButton.Add_Click({ & $openFolder $maskBox })
$maskClearButton.Add_Click({ $maskBox.Text = "" })
$outputButton.Add_Click({ & $openFolder $outputBox })

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 250
$script:augmentationProcess = $null
$timer.Add_Tick({
    if ($null -eq $script:augmentationProcess -or -not $script:augmentationProcess.HasExited) { return }
    $timer.Stop()
    $stdout = $script:augmentationProcess.StandardOutput.ReadToEnd()
    $stderr = $script:augmentationProcess.StandardError.ReadToEnd()
    if ($stdout) { $logBox.AppendText($stdout + [Environment]::NewLine) }
    if ($stderr) { $logBox.AppendText($stderr + [Environment]::NewLine) }
    $exitCode = $script:augmentationProcess.ExitCode
    $script:augmentationProcess.Dispose()
    $script:augmentationProcess = $null
    $runButton.Enabled = $true
    if ($exitCode -eq 0) {
        $statusLabel.Text = "Completed successfully"
        [System.Windows.Forms.MessageBox]::Show("Augmentation completed.", "Complete", "OK", "Information") | Out-Null
    } else {
        $statusLabel.Text = "Failed - see log"
        [System.Windows.Forms.MessageBox]::Show("Augmentation failed. See the log for details.", "Error", "OK", "Error") | Out-Null
    }
})

$runButton.Add_Click({
    try {
        if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Python executable not found: $pythonPath" }
        if (-not (Test-Path -LiteralPath $augmenterPath -PathType Leaf)) { throw "Augmenter script not found: $augmenterPath" }
        if (-not (Test-Path -LiteralPath $inputBox.Text)) { throw "Select an existing image file or folder." }
        if ([string]::IsNullOrWhiteSpace($outputBox.Text)) { throw "Select an output folder." }
        [int] $shiftPixels = 0
        [double] $smallAngle = 0
        [double] $fillValue = 0
        if ($shiftCheck.Checked -and -not [int]::TryParse($shiftBox.Text, [ref] $shiftPixels)) { throw "Translation must be an integer." }
        if ($smallRotationCheck.Checked -and -not [double]::TryParse($angleBox.Text, [ref] $smallAngle)) { throw "Small rotation must be a number." }
        if (-not [double]::TryParse($fillValueBox.Text, [ref] $fillValue)) { throw "Fill value must be a number." }

        $arguments = @($augmenterPath, $inputBox.Text, $outputBox.Text)
        if ($recursiveCheck.Checked) { $arguments += "--recursive" }
        if ($overwriteCheck.Checked) { $arguments += "--overwrite" }
        if (-not $quarterCheck.Checked) { $arguments += "--no-quarter-turns" }
        if (-not $flipCheck.Checked) { $arguments += "--no-flips" }
        if ($shiftCheck.Checked -and $shiftPixels -gt 0) { $arguments += @("--shift-pixels", [string] $shiftPixels) }
        if ($smallRotationCheck.Checked -and $smallAngle -ne 0) {
            $angle = [Math]::Abs($smallAngle).ToString([System.Globalization.CultureInfo]::InvariantCulture)
            $arguments += @("--small-angles", "-$angle", $angle)
        }
        if (-not [string]::IsNullOrWhiteSpace($maskBox.Text)) {
            if (-not (Test-Path -LiteralPath $maskBox.Text)) { throw "Mask path does not exist." }
            $arguments += @("--mask-input", $maskBox.Text)
        }
        $fillMode = if ($fillModeBox.SelectedIndex -eq 0) { "border-median" } else { "constant" }
        $arguments += @(
            "--fill-mode", $fillMode,
            "--fill-value", $fillValue.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )

        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $pythonPath
        $info.Arguments = (($arguments | ForEach-Object { Quote-Argument ([string] $_) }) -join " ")
        $info.WorkingDirectory = $projectDir
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true

        $logBox.Clear()
        $logBox.AppendText("Starting augmentation..." + [Environment]::NewLine)
        $statusLabel.Text = "Running..."
        $runButton.Enabled = $false
        $script:augmentationProcess = New-Object System.Diagnostics.Process
        $script:augmentationProcess.StartInfo = $info
        if (-not $script:augmentationProcess.Start()) { throw "Could not start the augmentation process." }
        $timer.Start()
    } catch {
        $runButton.Enabled = $true
        $statusLabel.Text = "Error"
        [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Cannot start", "OK", "Error") | Out-Null
    }
})

$form.Add_FormClosing({
    if ($null -ne $script:augmentationProcess -and -not $script:augmentationProcess.HasExited) {
        $_.Cancel = $true
        [System.Windows.Forms.MessageBox]::Show("Augmentation is still running. Wait for it to finish before closing.", "Running", "OK", "Warning") | Out-Null
    }
})

[void] $form.ShowDialog()
