<#
Copyright (c) Stefano Sinigardi

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
#>

$utils_psm1_version = "1.2.3"
$IsWindowsPowerShell = switch ( $PSVersionTable.PSVersion.Major ) {
  5 { $true }
  4 { $true }
  3 { $true }
  2 { $true }
  default { $false }
}

$ExecutableSuffix = ""
if ($IsWindowsPowerShell -or $IsWindows) {
  $ExecutableSuffix = ".exe"
}

$64bitPwsh = $([Environment]::Is64BitProcess)
$64bitOS = $([Environment]::Is64BitOperatingSystem)

Push-Location $PSScriptRoot
$GIT_EXE = Get-Command "git" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
if ($GIT_EXE) {
  $IsInGitSubmoduleString = $(git rev-parse --show-superproject-working-tree 2> $null)
  if ($IsInGitSubmoduleString.Length -eq 0) {
    $IsInGitSubmodule = $false
  }
  else {
    $IsInGitSubmodule = $true
  }
}
else {
  $IsInGitSubmodule = $false
}
Pop-Location

$cuda_version_full = "12.2.0"
$cuda_version_short = "12.2"
$cuda_version_full_dashed = $cuda_version_full.replace('.', '-')
$cuda_version_short_dashed = $cuda_version_short.replace('.', '-')

function getProgramFiles32bit() {
  $out = ${env:PROGRAMFILES(X86)}
  if ($null -eq $out) {
    $out = ${env:PROGRAMFILES}
  }

  if ($null -eq $out) {
    MyThrow("Could not find [Program Files 32-bit]")
  }

  return $out
}

function getLatestVisualStudioWithDesktopWorkloadPath([bool]$required = $true) {
  $programFiles = getProgramFiles32bit
  $vswhereExe = "$programFiles\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhereExe) {
    $output = & $vswhereExe -products * -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -format xml
    [xml]$asXml = $output
    foreach ($instance in $asXml.instances.instance) {
      $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
    }
    if (!$installationPath) {
      #Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also partial installations" -ForegroundColor Yellow
      $output = & $vswhereExe -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
      }
    }
    if (!$installationPath) {
      #Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also pre-release installations" -ForegroundColor Yellow
      $output = & $vswhereExe -prerelease -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
      }
    }
    if (!$installationPath) {
      if ($required) {
        MyThrow("Could not locate any installation of Visual Studio")
      }
      else {
        Write-Host "Could not locate any installation of Visual Studio" -ForegroundColor Red
        return $null
      }
    }
  }
  else {
    if ($required) {
      MyThrow("Could not locate vswhere at $vswhereExe")
    }
    else {
      Write-Host "Could not locate vswhere at $vswhereExe" -ForegroundColor Red
      return $null
    }
  }
  return $installationPath
}

function getLatestVisualStudioWithDesktopWorkloadVersion([bool]$required = $true) {
  $programFiles = getProgramFiles32bit
  $vswhereExe = "$programFiles\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhereExe) {
    $output = & $vswhereExe -products * -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -format xml
    [xml]$asXml = $output
    foreach ($instance in $asXml.instances.instance) {
      $installationVersion = $instance.InstallationVersion
    }
    if (!$installationVersion) {
      #Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also partial installations" -ForegroundColor Yellow
      $output = & $vswhereExe -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationVersion = $instance.installationVersion
      }
    }
    if (!$installationVersion) {
      #Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also pre-release installations" -ForegroundColor Yellow
      $output = & $vswhereExe -prerelease -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationVersion = $instance.installationVersion
      }
    }
    if (!$installationVersion) {
      if ($required) {
        MyThrow("Could not locate any installation of Visual Studio")
      }
      else {
        Write-Host "Could not locate any installation of Visual Studio" -ForegroundColor Red
        return $null
      }
    }
  }
  else {
    if ($required) {
      MyThrow("Could not locate vswhere at $vswhereExe")
    }
    else {
      Write-Host "Could not locate vswhere at $vswhereExe" -ForegroundColor Red
      return $null
    }
  }
  return $installationVersion
}

function setupVisualStudio([bool]$required = $true) {
  $CL_EXE = Get-Command "cl" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Definition
  if ((-Not $CL_EXE) -or ($CL_EXE -match "HostX86\\x86") -or ($CL_EXE -match "HostX64\\x86")) {
    $vsfound = getLatestVisualStudioWithDesktopWorkloadPath
    Write-Host "Found VS in ${vsfound}"
    Push-Location "${vsfound}/Common7/Tools"
    cmd.exe /c "VsDevCmd.bat -arch=x64 & set" |
    ForEach-Object {
      if ($_ -match "=") {
        $v = $_.split("="); Set-Item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
      }
    }
    Pop-Location
    $env:PATH = "${vsfound}/VC/Tools/Llvm/x64/bin;$env:PATH"
    Write-Host "Visual Studio Command Prompt variables set"
  }
}

function DownloadNinja() {
  Write-Host "Downloading a portable version of Ninja" -ForegroundColor Yellow
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue ninja
  Remove-Item -Force -ErrorAction SilentlyContinue ninja.zip
  if ($IsWindows -or $IsWindowsPowerShell) {
    $url = "https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-win.zip"
  }
  elseif ($IsLinux) {
    $url = "https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip"
  }
  elseif ($IsMacOS) {
    $url = "https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-mac.zip"
  }
  else {
    MyThrow("Unknown OS, unsupported")
  }
  Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile "ninja.zip"
  Expand-Archive -Path ninja.zip
  Remove-Item -Force -ErrorAction SilentlyContinue ninja.zip
  return "./ninja${ExecutableSuffix}"
}

function DownloadAria2() {
  Write-Host "Downloading a portable version of Aria2" -ForegroundColor Yellow
  if ($IsWindows -or $IsWindowsPowerShell) {
    $basename = "aria2-1.37.0-win-32bit-build1"
    $zipName = "${basename}.zip"
    $outFolder = "$basename/$basename"
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $outFolder
    Remove-Item -Force -ErrorAction SilentlyContinue $zipName
    $url = "https://github.com/aria2/aria2/releases/download/release-1.37.0/$zipName"
    Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile $zipName
    Expand-Archive -Path $zipName
  }
  elseif ($IsLinux) {
    $basename = "aria2-1.36.0-linux-gnu-64bit-build1"
    $zipName = "${basename}.tar.bz2"
    $outFolder = $basename
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $outFolder
    Remove-Item -Force -ErrorAction SilentlyContinue $zipName
    $url = "https://github.com/q3aql/aria2-static-builds/releases/download/v1.36.0/$zipName"
    Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile $zipName
    tar xf $zipName
  }
  elseif ($IsMacOS) {
    $basename = "aria2-1.35.0-osx-darwin"
    $zipName = "${basename}.tar.bz2"
    $outFolder = "aria2-1.35.0/bin"
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $outFolder
    Remove-Item -Force -ErrorAction SilentlyContinue $zipName
    $url = "https://github.com/aria2/aria2/releases/download/release-1.35.0/$zipName"
    Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile $zipName
    tar xf $zipName
  }
  else {
    MyThrow("Unknown OS, unsupported")
  }
  Remove-Item -Force -ErrorAction SilentlyContinue $zipName
  return "./$outFolder/aria2c${ExecutableSuffix}"
}

function DownloadLicencpp() {
  $licencpp_version = "0.2.1"
  Write-Host "Downloading a portable version of licencpp v${licencpp_version}" -ForegroundColor Yellow
  if ($IsWindows -or $IsWindowsPowerShell) {
    $basename = "licencpp-Windows"
  }
  elseif ($IsLinux) {
    $basename = "licencpp-Linux"
  }
  else {
    MyThrow("Unknown OS, unsupported")
  }
  $zipName = "${basename}.zip"
  $outFolder = "${basename}"
  Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $outFolder
  Remove-Item -Force -ErrorAction SilentlyContinue $zipName
  $url = "https://github.com/cenit/licencpp/releases/download/v${licencpp_version}/$zipName"
  Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile $zipName
  Expand-Archive -Path $zipName
  Remove-Item -Force -ErrorAction SilentlyContinue $zipName
  return "./$outFolder/licencpp${ExecutableSuffix}"
}

function Download7Zip() {
  Write-Host "Downloading a portable version of 7-Zip" -ForegroundColor Yellow
  if ($IsWindows -or $IsWindowsPowerShell) {
    $basename = "7za920"
    $zipName = "${basename}.zip"
    $outFolder = "$basename"
    $outSuffix = "a"
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $outFolder
    Remove-Item -Force -ErrorAction SilentlyContinue $zipName
    $url = "https://www.7-zip.org/a/$zipName"
    Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile $zipName
    Expand-Archive -Path $zipName
  }
  elseif ($IsLinux) {
    $basename = "7z2201-linux-x64"
    $zipName = "${basename}.tar.xz"
    $outFolder = $basename
    $outSuffix = "z"
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $outFolder
    Remove-Item -Force -ErrorAction SilentlyContinue $zipName
    $url = "https://www.7-zip.org/a/$zipName"
    Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile $zipName
    tar xf $zipName
  }
  elseif ($IsMacOS) {
    $basename = "7z2107-mac"
    $zipName = "${basename}.tar.xz"
    $outFolder = $basename
    $outSuffix = "z"
    Remove-Item -Force -Recurse -ErrorAction SilentlyContinue $outFolder
    Remove-Item -Force -ErrorAction SilentlyContinue $zipName
    $url = "https://www.7-zip.org/a/$zipName"
    Invoke-RestMethod -Uri $url -Method Get -ContentType application/zip -OutFile $zipName
    tar xf $zipName
  }
  else {
    MyThrow("Unknown OS, unsupported")
  }
  Remove-Item -Force -ErrorAction SilentlyContinue $zipName
  return "./$outFolder/7z${outSuffix}${ExecutableSuffix}"
}

Function MyThrow ($Message) {
  if ($global:DisableInteractive) {
    Write-Host $Message -ForegroundColor Red
    throw
  }
  else {
    # Check if running in PowerShell ISE
    if ($psISE) {
      # "ReadKey" not supported in PowerShell ISE.
      # Show MessageBox UI
      $Shell = New-Object -ComObject "WScript.Shell"
      $Shell.Popup($Message, 0, "OK", 0)
      throw
    }

    $Ignore =
    16, # Shift (left or right)
    17, # Ctrl (left or right)
    18, # Alt (left or right)
    20, # Caps lock
    91, # Windows key (left)
    92, # Windows key (right)
    93, # Menu key
    144, # Num lock
    145, # Scroll lock
    166, # Back
    167, # Forward
    168, # Refresh
    169, # Stop
    170, # Search
    171, # Favorites
    172, # Start/Home
    173, # Mute
    174, # Volume Down
    175, # Volume Up
    176, # Next Track
    177, # Previous Track
    178, # Stop Media
    179, # Play
    180, # Mail
    181, # Select Media
    182, # Application 1
    183  # Application 2

    Write-Host $Message -ForegroundColor Red
    Write-Host -NoNewline "Press any key to continue..."
    while (($null -eq $KeyInfo.VirtualKeyCode) -or ($Ignore -contains $KeyInfo.VirtualKeyCode)) {
      $KeyInfo = $Host.UI.RawUI.ReadKey("NoEcho, IncludeKeyDown")
    }
    Write-Host ""
    throw
  }
}

Function CopyTexFile ($MyFile) {
  $MyFileName = Split-Path $MyFile -Leaf
  New-Item -ItemType Directory -Force -Path "~/${latex_path}" | Out-Null
  if (-Not (Test-Path "~/${latex_path}/$MyFileName" )) {
    Write-Host "Copying $MyFile to ~/${latex_path}"
    Copy-Item "$MyFile" "~/${latex_path}"
  }
  else {
    Write-Host "~/${latex_path}/$MyFileName already present"
  }
}

Function dos2unix {
  Param (
    [Parameter(mandatory = $true)]
    [string[]]$path
  )

  Get-ChildItem -File -Recurse -Path $path |
  ForEach-Object {
    Write-Host "Converting $_"
    $x = get-content -raw -path $_.fullname; $x -replace "`r`n", "`n" | Set-Content -NoNewline -Force -path $_.fullname
  }
}

Function unix2dos {
  Param (
    [Parameter(mandatory = $true)]
    [string[]]$path
  )

  Get-ChildItem -File -Recurse -Path $path |
  ForEach-Object {
    $x = get-content -raw -path $_.fullname
    $SearchStr = [regex]::Escape("`r`n")
    $SEL = Select-String -InputObject $x -Pattern $SearchStr
    if ($null -ne $SEL) {
      Write-Host "Converting $_"
      # do nothing: avoid creating files containing `r`r`n when using unix2dos twice on the same file
    }
    else {
      Write-Host "Converting $_"
      $x -replace "`n", "`r`n" | Set-Content -NoNewline -Force -path $_.fullname
    }
  }
}

Function UpdateRepo {
  if ($GIT_EXE) {
    Get-ChildItem -Directory |
      ForEach-Object {
      Set-Location $_.Name
      git pull
      git submodule update --recursive
      Set-Location ..
    }
  }
}

Export-ModuleMember -Variable utils_psm1_version
Export-ModuleMember -Variable IsWindowsPowerShell
Export-ModuleMember -Variable IsInGitSubmodule
Export-ModuleMember -Variable 64bitPwsh
Export-ModuleMember -Variable 64bitOS
Export-ModuleMember -Variable cuda_version_full
Export-ModuleMember -Variable cuda_version_short
Export-ModuleMember -Variable cuda_version_full_dashed
Export-ModuleMember -Variable cuda_version_short_dashed
Export-ModuleMember -Function getProgramFiles32bit
Export-ModuleMember -Function getLatestVisualStudioWithDesktopWorkloadPath
Export-ModuleMember -Function getLatestVisualStudioWithDesktopWorkloadVersion
Export-ModuleMember -Function setupVisualStudio
Export-ModuleMember -Function DownloadNinja
Export-ModuleMember -Function DownloadAria2
Export-ModuleMember -Function Download7Zip
Export-ModuleMember -Function DownloadLicencpp
Export-ModuleMember -Function MyThrow
Export-ModuleMember -Function CopyTexFile
Export-ModuleMember -Function dos2unix
Export-ModuleMember -Function unix2dos
Export-ModuleMember -Function UpdateRepo
