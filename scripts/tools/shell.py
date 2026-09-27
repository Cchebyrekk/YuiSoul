# -*- coding: utf-8 -*-
"""
Команды PowerShell для Юи. Всё, что может что-то изменить (удалить, запустить, установить, записать,
поменять настройки или сеть), выполняется только после подтверждения пользователя — см. approval.py.
Проверка намеренно подозрительная: лишний вопрос дешевле, чем удалённая папка.
"""
import re
import subprocess
from typing import Callable, Optional

COMMAND_TIMEOUT = 60
OUTPUT_MAX_CHARS = 4000

# Глаголы PowerShell, которые меняют систему (Remove-Item, Set-ItemProperty, Start-Process, Out-File...)
_CHANGING_VERBS = (
    "remove|clear|set|stop|restart|format|uninstall|install|new|add|rename|move|copy|disable|enable|invoke|"
    "start|suspend|resume|mount|dismount|initialize|reset|update|register|unregister|import|export|write|out|"
    "send|lock|unlock|grant|revoke|block|unblock|protect|unprotect|publish|restore|backup|checkpoint|join|save|"
    "sync|connect|disconnect|enter|expand|compress|edit|push|pop|merge|split"
)
_CHANGING_CMDLET_RE = re.compile(rf"\b(?:{_CHANGING_VERBS})-[a-z]+", re.IGNORECASE)
# Из них безвредные: вывод на экран, форматирование вывода (но не Format-Volume!) и смена текущей папки
_HARMLESS_CMDLETS = {"out-string", "out-host", "out-null", "write-output", "write-host", "write-verbose",
                     "write-information", "set-location", "push-location", "pop-location",
                     "format-list", "format-table", "format-wide", "format-custom", "format-hex"}
# Короткие псевдонимы и программы, которые что-то меняют или запускают произвольный код
_CHANGING_WORDS = {
    "rm", "del", "erase", "rd", "rmdir", "ri", "mv", "move", "mi", "ren", "rni", "cp", "copy", "cpi", "md", "mkdir",
    "ni", "sc", "si", "sp", "kill", "spps", "taskkill", "shutdown", "logoff", "format", "diskpart", "bcdedit", "reg",
    "regedit", "net", "netsh", "schtasks", "takeown", "icacls", "cacls", "cipher", "vssadmin", "wmic", "powercfg",
    "winget", "choco", "scoop", "pip", "pip3", "npm", "msiexec", "curl", "wget", "iwr", "irm", "iex", "icm", "saps",
    "start", "attrib", "robocopy", "xcopy", "setx", "sfc", "dism", "mountvol", "label", "defrag", "chkdsk", "runas",
    "certutil", "bitsadmin", "mshta", "rundll32", "regsvr32", "cscript", "wscript", "python", "py", "node", "cmd",
    "powershell", "pwsh", "git", "ii", "sal", "nal", "ipmo",
}


def confirmation_reason(command: str) -> Optional[str]:
    """Почему команде нужно подтверждение (None — только читает/показывает)."""
    for m in _CHANGING_CMDLET_RE.finditer(command):
        if m.group(0).lower() not in _HARMLESS_CMDLETS:
            return f"меняет систему ({m.group(0)})"
    # Слова целиком, с дефисом («format-list» — не «format»), без параметров («Get-Date -Format» — не format)
    for word in re.findall(r"(?<![\w-])[a-z_][\w.-]*", command.lower()):
        if re.sub(r"\.(exe|com|bat|cmd|ps1|msi|vbs)$", "", word) in _CHANGING_WORDS or word.endswith((".bat", ".cmd", ".ps1", ".msi", ".vbs")):
            return f"меняет систему или запускает программу ({word})"
    if ">" in command:
        return "пишет в файл (>)"
    if re.search(r"(^|[;|{(]\s*)[&.]\s*[\"'$\w\\]", command):
        return "запускает скрипт или программу (& / .)"
    return None


def run_command(command: str, approve: Callable[[str], bool]) -> str:
    command = (command or "").strip()
    if not command:
        return "Ошибка: пустая команда."
    reason = confirmation_reason(command)
    if reason:
        if not approve(f"Юи хочет выполнить команду PowerShell ({reason}):\n\n{command}"):
            return ("Пользователь НЕ разрешил эту команду (или не ответил за 2 минуты) — она не выполнена. "
                    "Не пытайся обойти запрет другой командой; спроси его, как быть.")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; " + command],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=COMMAND_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return f"Команда не завершилась за {COMMAND_TIMEOUT} с и была остановлена."
    except Exception as e:
        return f"Ошибка запуска PowerShell: {e}"
    output = (result.stdout + ("\n" + result.stderr if result.stderr.strip() else "")).strip() or "(пустой вывод)"
    if len(output) > OUTPUT_MAX_CHARS:
        output = output[:OUTPUT_MAX_CHARS] + f"\n[... обрезано, всего {len(output)} символов]"
    # Вывод — данные (содержимое файлов, страниц), а не указания для Юи
    return f"Код выхода {result.returncode}.\n<command_output>\n{output}\n</command_output>"
