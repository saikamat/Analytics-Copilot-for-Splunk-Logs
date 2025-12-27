import random
from datetime import datetime, timedelta

sources = ['sshd', 'kernel', 'systemd', 'nginx', 'postgresql', 'docker', 'cron', 'auth', 'networking']
log_levels = ['INFO', 'WARN', 'ERROR']
#  = ['server1', 'server2', 'server3', 'db1', 'web1', 'cache1']
# messages = [
#     'User login successful',
#     'Disk space running low',
#     'Service started',
#     'Connection timed out',
#     'File not found',
#     'Permission denied',
#     'Database connection established',
#     'Configuration file updated',
#     'Memory usage high',
#     'Process terminated unexpectedly'
# ]
metadata = {
    'sshd': [
        {'user': 'alice', 'ip': '192.168.1.100'},
        {'user': 'bob', 'ip': '192.168.1.101'},
        {'user': 'charlie', 'ip': '192.168.1.102'},
    ],
    'auth': [
        {'user': 'alice', 'ip': '192.168.1.100'},
        {'user': 'diana', 'ip': '192.168.1.103'},
        {'user': 'eve', 'ip': '192.168.1.104'},
    ],
    'kernel': [{'process': 'systemd', 'pid': 1}],
    'systemd': [{'service': 'nginx'}, {'service': 'postgresql'}, {'service': 'docker'}],
    'nginx': [{'status': '200'}, {'status': '500'}, {'status': '503'}],
    'postgresql': [{'database': 'main'}, {'database': 'cache'}],
    'docker': [{'container': 'web'}, {'container': 'db'}],
    'cron': [{'job': 'backup'}, {'job': 'cleanup'}],
    'networking': [{'interface': 'eth0'}, {'interface': 'eth1'}]
}

def timestamp_generator():
    base = datetime.now()
    for _ in range(1000):
        ts = base - timedelta(minutes=random.randint(0, 10000))
        yield ts

def generate_logs(count: int) -> list[dict]:
    source_messages = {
        'sshd': ['User login successful', 'User login failed', 'Connection closed'],
        'kernel': ['Disk space running low', 'Memory usage high', 'Process terminated unexpectedly'],
        'systemd': ['Service started', 'Service stopped', 'Configuration file updated'],
        'nginx': ['Connection timed out', 'Request processed', 'Bad gateway'],
        'postgresql': ['Database connection established', 'Query timeout', 'Connection refused'],
        'docker': ['Container started', 'Container stopped', 'Image pulled'],
        'cron': ['Job executed', 'Job failed', 'Schedule updated'],
        'auth': ['User login successful', 'Permission denied', 'Authentication failed'],
        'networking': ['Connection established', 'Packet loss detected', 'DNS resolution failed']
    }
    
    logs = []
    ts_gen = timestamp_generator()
    for _ in range(count):
        source = random.choice(sources)
        log_entry = {
            'timestamp': next(ts_gen),
            'level': random.choices(log_levels, weights=[70, 20, 10], k=1)[0],
            'source': source,
            'message': random.choice(source_messages[source]),
            'metadata': random.choice(metadata[source])
        }
        logs.append(log_entry)
    return logs

if __name__ == "__main__":
    sample_logs = generate_logs(10)
    for log in sample_logs:
        print(log)


