import re
from datetime import datetime
from datetime import timedelta
from datetime import timezone

SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)

def getNowTime():
    # 协调世界时
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)

    # 北京时间
    beijing_now = str(utc_now.astimezone(SHA_TZ))
    now_time_str=re.sub(":|-| ","_",beijing_now)
    # print(beijing_now)
    # print(now_time_str)
    return now_time_str


if __name__ == '__main__':
    print(getNowTime())