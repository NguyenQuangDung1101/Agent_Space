from datetime import datetime


def get_current_datetime():
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    weekday_str = now.strftime('%A')
    
    output = f"Current date: {date_str} - Day: {weekday_str} - Current time: {time_str}"
    return output