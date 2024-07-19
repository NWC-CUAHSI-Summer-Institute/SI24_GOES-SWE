#!/usr/bin/env python3

"""
This is a collection of functions for making it easier to work with the
CUAHSI HIS in the Python 3 language.

Written By: Tony Castronova <acastronova@cuahsi.org>
Last modified: 08.26.2019

"""

import funcs
import numpy
import pandas
from itertools import tee
from suds.client import Client
from multiprocessing import Pool, cpu_count


class Services(object):
    def __init__(self):
        self.wsdl = 'https://hiscentral.cuahsi.org/webservices/' + \
                    'hiscentral.asmx?wsdl'
        self.client = Client(self.wsdl)
        self.utils = Utilities(self.client)

    def get_data_providers(self):
        # collect service info via GetWaterOneFlowServiceInfo
        service_info = self.client.service.GetWaterOneFlowServiceInfo()

        # convert into a pandas dataframe
        dat = []
        for si in service_info[0]:
            dat.append(dict(si))
        return pandas.DataFrame(dat)

    def get_sites(self, xmin, ymin, xmax, ymax,
                  pattern=None, degStep=None, **kwargs):
        """
        xmin: minimum x coordinate of search box
        ymin: minimum y coordinate of search box
        xmax: maximum x coordinate of search box
        ymax: maximum y coordinate of search box
        pattern: search pattern for data service network name, e.g. NWIS
        degStep: degree value to split bounding box into, necessary for large
                 searches since the API will not return values over 25k
        **kwargs: additional keyword arguments
                - conceptKeyword: concept keyword to search, e.g. surface water
                - networkIDs: provider IDs to search (list)
        """

        keyword = kwargs.get('conceptKeyword', '')
        ids = ','.join(kwargs.get('networkIDs', []))

        # get all services (i.e. data providers)
        data_services = self.get_data_providers()

        # filter by pattern if provided
        if pattern:
            data_services = data_services[data_services['NetworkName'].
                                          str.contains(pattern)]

        # remove unnecessary columns
        service_ids = data_services[['NetworkName',
                                     'ServiceDescriptionURL',
                                     'ServiceID']]

        # subset bbox if specified
        if degStep is not None:
            subset = self.utils.subset_bounding_box(xmin,
                                                    ymin,
                                                    xmax,
                                                    ymax,
                                                    degStep)
            x_coords = subset['xcoords']
            y_coords = subset['ycoords']
        else:
            x_coords = [(xmin, xmax)]
            y_coords = [(ymin, ymax)]

        # combine input ids with ids determined via pattern searching
        serv_ids = list(service_ids['ServiceID'])
        serv_ids.extend(ids)

        # loop over each search region defined in x and y coords lists
        # and build parameter sets
        parameters = []
        for sid in serv_ids:
            for x in x_coords:
                for y in y_coords:
                    # build search parameter dictionary
                    params = dict(ymin=y[0],
                                  xmin=x[0],
                                  ymax=y[1],
                                  xmax=x[1],
                                  conceptKeyword=keyword,
                                  networkIDs=sid)
                    parameters.append(params)

        print(f'number of param sets: {len(parameters)}')

        # run get sites in parallel
        pool = Pool(cpu_count())
        args = [(self.wsdl, p) for p in parameters]
        result = pool.starmap(funcs.get_sites, args)

        # convert result into a pandas dataframe
        print('converting to pandas dataframe')
        dat = []
        for r in result:
            for site in r:
                dat.append(dict(site))
        return pandas.DataFrame(dat)

    def get_sites_info(self, wsdl_list, siteid_list, verbose=False):
        """
        wsdl: SOAP endpoint for the desired service, e.g.
              http://hydroportal.cuahsi.org/nwisdv/cuahsi_1_1.asmx?wsdl
        site: the site to query in the format NetworkName:SiteCode, e.g.
              NWISDV:12120005
        """

        data = []
        for i in range(len(wsdl_list)):
            wsdl = wsdl_list[i]
            if wsdl[-5:] != '?wsdl':
                wsdl = f'{wsdl}?wsdl'
            siteid = siteid_list[i]

            if verbose:
                print(f'+ processing site [{i} of {len(wsdl_list)}] -> ' +
                      f'{siteid}')

            # create new instance of soap class
            client = Client(wsdl)

            # run the SOAP query
            result = client.service.GetSiteInfoObject(siteid)
            
            for site in result.site:
                for cat in site.seriesCatalog:
                    for info in cat.series:
                        #import pdb; pdb.set_trace()
                        
                        dat = dict(method=info.method.methodDescription,
                                   variableName=info.variable.variableName,
                                   variableCode=info.variable.variableCode[0].value,
                                   datatype=info.variable.dataType,
                                   valuetype=info.variable.valueType,
                                   sampleMedium=info.variable.sampleMedium,
                                   startdt=info.variableTimeInterval.beginDateTime,
                                   enddt=info.variableTimeInterval.endDateTime,
                                   name=site.siteInfo.siteName,
                                   lat=site.siteInfo.geoLocation.geogLocation.latitude,
                                   lon=site.siteInfo.geoLocation.geogLocation.longitude,
                                   siteid=siteid)
                        data.append(dat)

        if verbose:
            print(f'+ building pandas object')
        return pandas.DataFrame(data)



class Utilities(object):
    def __init__(self, client):
        self.client = client

    def pairwise(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return list(zip(a, b))

    def subset_bounding_box(self, xmin, ymin, xmax, ymax, degStep):

        # split ranges into subranges by degree step
        xrange = numpy.arange(xmin, xmax, degStep)
        yrange = numpy.arange(ymin, ymax, degStep)

        # split ranges into min,max subrange tuples
        x_coords = self.pairwise(xrange)
        y_coords = self.pairwise(yrange)

        # return dictionary of x and y coordinates
        return dict(xcoords=x_coords,
                    ycoords=y_coords)
