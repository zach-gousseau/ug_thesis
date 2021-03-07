from datetime import datetime, date, timedelta

def subtract_times(t2, t1, day_delta='auto'):
    t1_day = date.today()
    t2_day = date.today()
    if day_delta == 'auto':
        if t2 < t1:
            t2_day += timedelta(days=1)
    elif day_delta == 'off':
        pass
    elif type(day_delta) == int:
        t2.day += timedelta(days=day_delta)
    else:
        raise ValueError('day_delta must be one of: auto, off or an integer')
    return datetime.combine(t2_day, t2) - datetime.combine(t1_day,  t1)