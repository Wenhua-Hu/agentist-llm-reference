"""Create synthetic demo tenants through the same admin API; no embedded secrets."""
import argparse
import os
import httpx

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    key=os.environ.get('ADMIN_API_KEY')
    if not key: parser.error('ADMIN_API_KEY required in environment')
    with httpx.Client(base_url=args.base_url,headers={'Authorization':'Bearer '+key},trust_env=False,timeout=10) as client:
        for name in ('Alice demo','Bob demo'):
            response=client.post('/admin/tenants',json={'name':name,'balance_micro_usd':1000000,
                'token_limit':100000,'rpm_limit':60,'concurrency_limit':4})
            response.raise_for_status()
            print(name+': '+response.json()['tenant_id'])
    print('Open /console to issue keys; this script does not print or persist credentials.')
if __name__=='__main__':main()
