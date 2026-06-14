from datetime import timedelta


def format_time(td: timedelta) -> str:
    seconds = int(td.total_seconds())
    periods = [
        ('day', 86400),
        ('hour', 3600),
        ('min', 60),
        ('sec', 1)
    ]

    for name, count in periods:
        if seconds >= count:
            value = seconds // count
            suffix = 's' if value > 1 and name in ['hour', 'day'] else ''
            return f"{value} {name}{suffix}"

    return "0 sec"