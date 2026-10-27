import subprocess
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")

RUN_HOUR = 10
RUN_MINUTE = 0
CHECK_INTERVAL = 20  # seconds between checks

THEORIES = [
    ("RSP BREADTH", ["python", "-m", "theories.rsp_breadth", "--save", "--telegram"]),
    ("DURATION / VOLATILITY", ["python", "-m", "theories.duration_volatility", "--save", "--telegram"]),
    ("MACRO STRESS", ["python", "-m", "theories.macro_stress", "--save", "--telegram"]),
    ("BTC MOMENTUM", ["python", "theories/daily_strategy_check.py"]),
]


def run_theory(name, command):
    """Runs a single theory and returns its exit code."""
    print()
    print("=" * 70)
    print("RUNNING:", name)
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("=" * 70)

    print("COMMAND:", " ".join(command))
    print()

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd='/app',  # working directory
        )

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print()
            print("SUCCESS:", name)
        else:
            print()
            print("FAILED:", name)
            print("EXIT CODE:", result.returncode)

        return result.returncode

    except Exception as exc:
        print()
        print("ERROR:", name)
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception message: {exc}")
        return -1


def run_all_theories():
    """Runs all theories and prints a summary of the results."""
    print()
    print("#" * 70)
    print("MARKET LAB - RUNNING ALL THEORIES")
    print(datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("#" * 70)

    results = []

    for name, command in THEORIES:
        code = run_theory(name, command)
        results.append((name, code))

    print()
    print("#" * 70)
    print("FINAL RESULTS")
    print("#" * 70)

    success_count = 0
    fail_count = 0

    for name, code in results:
        if code == 0:
            print(f"✅ OK  : {name}")
            success_count += 1
        else:
            print(f"❌ FAIL: {name} (EXIT CODE: {code})")
            fail_count += 1

    print("#" * 70)
    print(f"SUMMARY: {success_count} succeeded, {fail_count} failed")
    print("#" * 70)
    
    return fail_count == 0


def is_weekday(date):
    """Checks whether the day is a weekday (0-4 = Mon-Fri)."""
    return date.weekday() < 5


def should_run(now, last_run_date):
    """Decides whether the theories should run now."""
    if not is_weekday(now.date()):
        return False
    
    run_time = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    
    if now >= run_time and last_run_date != now.date():
        return True
    
    return False


def main():
    print()
    print("=" * 70)
    print("MARKET LAB SCHEDULER STARTED")
    print("=" * 70)
    
    print("Schedule: Monday-Friday at 10:00 America/New_York")
    print(f"Check interval: {CHECK_INTERVAL} seconds")
    print()
    print("THEORIES:")
    
    for name, command in THEORIES:
        print(f"  • {name} -> {' '.join(command)}")
    
    print("=" * 70)
    
    last_run_date = None
    
    while True:
        try:
            now = datetime.now(TZ)
            
            if should_run(now, last_run_date):
                print()
                print(f"🚀 Triggering scheduled run at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                
                success = run_all_theories()
                
                last_run_date = now.date()
                
                print()
                if success:
                    print("✅ All theories completed successfully.")
                else:
                    print("⚠️ Some theories failed. Check logs above.")
                
                next_run = now + timedelta(days=1)
                while not is_weekday(next_run.date()):
                    next_run += timedelta(days=1)
                
                next_run = next_run.replace(
                    hour=RUN_HOUR,
                    minute=RUN_MINUTE,
                    second=0,
                    microsecond=0
                )
                
                print(f"Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print()
            
            time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            print()
            print("🛑 Scheduler stopped by user.")
            break
        
        except Exception as exc:
            print()
            print("⚠️ SCHEDULER ERROR:")
            print(f"Exception type: {type(exc).__name__}")
            print(f"Exception message: {exc}")
            print("Continuing...")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()