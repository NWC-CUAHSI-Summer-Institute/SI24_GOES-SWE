#!/usr/bin/env python3

import os
from suds.client import Client


def get_sites(wsdl, params):

    param_str = f'[{params["xmin"]},{params["ymin"]},' + \
                f'{params["xmax"]},{params["ymax"]}] - ' + \
                f'{params["networkIDs"]} - {params["conceptKeyword"]}'
    print(f'process id: {os.getpid()} - {param_str} ', end='')
    try:
        client = Client(wsdl)
        f = client.factory.create('GetSitesInBox2')
        print('got sites')
        f.__dict__.update(params)
        result = client.service.GetSitesInBox2(f)
        print('got result')
        print(type(result))
        if str(type(result)) == "<class 'suds.sudsobject.ArrayOfSite'>":
            return [dict(r) for r in result[0]]
        else:
            return []
    except Exception as e:
        print(f'exception: {e}')
    print()
