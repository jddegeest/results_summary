import os
import shutil
from configparser import ConfigParser
import pandas as pd
import numpy as np
from glob import glob


def get_config():
    parser = ConfigParser()
    parser.read('results_summary.cfg')
    delete_temp_files = bool(parser.get('inputs', 'delete_temp_files'))
    base_name = parser.get('inputs', 'base_name')
    base_folder = parser.get('inputs', 'base_folder')
    base_ain = parser.get('inputs', 'base_ain')
    dev_name = parser.get('inputs', 'dev_name')
    dev_folder = parser.get('inputs', 'dev_folder')
    dev_ain = parser.get('inputs', 'dev_ain')
    file_dict = {base_name: (base_folder, base_ain),
                 dev_name: (dev_folder, dev_ain)}
    return file_dict, delete_temp_files


def calc_settings():
    settings = {'f133': {'variables': ['f133HostIF', 'f133VEDIF'],
                         'sign': None,
                         'calc_type': None,
                         'calc_var': None,
                         'res_file': 'SubTotalF133',
                         'filter_column': 'CohortKey',
                         'filter_list_type': 'cohort'},

                'dac': {'variables': ['AVIF',    # Ave * NIER
                                      'AOptValue',    # Diff
                                      'f97vEgpSurrChg',
                                      'fvrGMWBCharge', 'fvrGMDBCharge',
                                      'fvrStrategyChargeFromPrincipal',
                                      'AOptMVMat', 'AOptMVSold',
                                      'f97vEgpAdmExp',
                                      'fvrIntCred', 'fvrIntCredEOM',
                                      'ttrvfvrGMBCharge2',
                                      'ttrvfvrGMDBCharge2',
                                      'DBIntCredOnDeathPaid', 'WBReFeeGAAP',
                                      'AOptMVPur'],
                        'sign': [1] * 8 + [-1] * 8,
                        'calc_type': 'PV_egp',
                        'calc_var': ['PV_EGP'],
                        'res_file': 'SubTotalGAAP',
                        'filter_column': 'ReportingGroup',
                        'filter_list_type': 'cohort'},

                'sop_pre': {'variables': ['fvrGMDBClaim', 'fvrGMWBCredit',
                                          'fvrGMWBDBClaim'],
                            'sign': [1] * 3,
                            'calc_type': 'PV_claims',
                            'calc_var': ['SOP_Pre'],
                            'res_file': 'SubTotalSOP',
                            'filter_column': 'ReportingGroup',
                            'filter_list_type': 'cohort'},

                'sop_full': {'variables': ['fvrGMDBClaim', 'fvrGMWBCredit',
                                           'fvrGMWBDBClaim'],
                             'sign': [1] * 3,
                             'calc_type': 'PV_claims',
                             'calc_var': ['SOP_Full'],
                             'res_file': 'CyclicalSOP',
                             'filter_column': 'ReportingGroup',
                             'filter_list_type': 'cohort'},

                'stat': {'variables': {'cv_iapp': 'InitGuarCSVwoBAV',
                                       'cv_nopp': 'InitGuarCSVwBAV',
                                       'stat': 'TabResNoFloorStat_ts_E0',
                                       'tax': 'TabResNoFloorTax_ts_E0',
                                       'han_net': ('TabResNoFloorStatNetWBRe'
                                                   '_ts_E0'),
                                       'de_pct': 'ReinsDEpct',
                                       'wbre': 'WBRE_Cohort',
                                       'AG33apply': 'AG33apply',
                                       'av': 'AVIF_ts_E0',
                                       'av_floor': 'CARVMAnnBen93Floor',
                                       'AdminSystem': 'AdminSystem',
                                       'ss': ('ssNegWorstPVNetRevSer'
                                              '_ts_ELastValue')},
                         'sign': None,
                         'calc_type': 'stat',
                         'calc_var': ['STATRESERVE', 'TAXRESERVE',
                                      'DE_STATRESERVE'],
                         'res_file': 'InvStatProd',
                         'filter_column': 'Company',
                         'filter_list_type': 'company'},

                'payout': {'variables': ['GAAPf91BenResIF_ts_E0',
                                         'GAAPf60BenResIF_ts_E0',
                                         'GAAPf60MaintResIF_ts_E0',
                                         'GAAPf91ExpResIF_ts_E0',
                                         'TabRes_ts_E0', 'tax.TabRes_ts_E0',
                                         'UPR_th_E0'],
                           'sign': None,
                           'calc_type': None,
                           'calc_var': None,
                           'res_file': 'InvPayout',
                           'filter_column': 'Company',
                           'filter_list_type': 'company'},

                'rm': {'variables': ['TabRes', 'TotalResCeded', # time 0 only
                                     'PolLoan', # diff
                                     'AcqExp', 'AnnBenPaidBeg',
                                     'AnnBenPaidEnd', 'CashPartialW', 'Comm',
                                     'DeathBen', 'DivPaidOnLapse',
                                     'DivPaidPersist', 'MaintExp%',
                                     'MaintExpQ', 'MaintExpReserve',
                                     'MaintExpUnit', 'MaintExpW', 'SurrBen',
                                     'WBReFeeGAAP',
                                     'AOptMVPur',  # last positive
                                     'AOptMVMat', 'AOptMVSold',
                                     'CashIntOnPL', 'ChargeBackOnDeath',
                                     'ChargeBackOnSurr', 'ReinsClaims',
                                     'ReinsEA', 'ReinsSurrBen'],
                       'sign': [-1] * 2 + [1] * 17 + [-1] * 8,
                       'calc_type': 'rm',
                       'calc_var': ['PVLCF', 'WAL', 'COF'],
                       'res_file': 'SubTotalGAAP',
                       'filter_column': 'ReportingGroup',
                       'filter_list_type': 'cohort'},

                'nb_cof': {'variables': ['CashPrem',  # time 0 only
                                         'AcqExp', 'AnnBenPaidBeg',
                                         'AnnBenPaidEnd', 'CashPartialW',
                                         'Comm',
                                         'DeathBen', 'DivPaidOnLapse',
                                         'DivPaidPersist', 'MaintExp%',
                                         'MaintExpQ', 'MaintExpReserve',
                                         'MaintExpUnit', 'MaintExpW',
                                         'SurrBen',
                                         'WBReFeeGAAP',
                                         'AOptMVPur',  # last positive
                                         'AOptMVMat',
                                         'AOptMVSold',
                                         'CashIntOnPL', 'ChargeBackOnDeath',
                                         'ChargeBackOnSurr', 'ReinsClaims',
                                         'ReinsEA', 'ReinsSurrBen'],
                           'sign': [-1] + [1] * 16 + [-1] * 8,
                           'calc_type': 'nb_cof',
                           'calc_var': ['nb_cof'],
                           'res_file': 'Total001',
                           'filter_column': None,
                           'filter_list_type': None},

                'nb_res': {'variables': ['TabRes'],
                           'sign': None,
                           'calc_type': 'nb_res',
                           'calc_var': ['nb_res'],
                           'res_file': 'Total001',
                           'filter_column': None,
                           'filter_list_type': None}
                }

    return settings


def get_lob_settings_dict():
    settings = {'aaia': {'calcs': [('f133', 4), ('payout', 17),
                                   ('stat', None), ('dac', 1),
                                   ('sop_pre', 2), ('sop_full', 3),
                                   ('rm', 12), ('nb_cof', 501),
                                   ('nb_res', 502)],
                         'filter_type': 'drop',
                         'cohort_list': ['AADE_DD04_FIA_NOIR',
                                         'AADE_DD04_TDA_NOIR',
                                         'AADE_DD05_TDA_NOIR',
                                         'DA', 'EIA', 'EIX12', 'EIX13',
                                         'EIX14', 'EIX15', 'EIXACQI',
                                         'EIXACQM', 'EIXPE13', 'EIXPE14',
                                         'EIXPE15', 'GWB12', 'GWB13', 'GWB14',
                                         'GWB15', 'GWBACQ', 'GWBBEN12',
                                         'GWBBEN13', 'GWBBEN14', 'GWBBEN15',
                                         'MYGA', 'REG11', 'REG12', 'REG13',
                                         'REG14', 'REG15', 'REGACQ',
                                         'DD04_NON_2017', 'DD04_NON_2018',
                                         'DD04_NON_2019', 'VIAC_FIA_IR',
                                         'VIAC_FIA_NON', 'VIAC_GMIB_LC_SC',
                                         'VIAC_GMIB_NLC_SC', 'VIAC_PAYOUT_LC',
                                         'VIAC_PAYOUT_LC_SC',
                                         'VIAC_PAYOUT_NLC',
                                         'VIAC_PAYOUT_NLC_SC', 'VIAC_SA',
                                         'VIAC_TDA_NOIR_AR',
                                         'VIAC_TDA_NOIR_MYGA', 'RLI_FIA_IR',
                                         'RLI_FIA_NON', 'RLI_PAYOUT_LC',
                                         'RLI_PAYOUT_LC_SC', 'RLI_PAYOUT_NLC',
                                         'RLI_PAYOUT_NLC_SC', 'RLI_SA',
                                         'RLI_TDA_NOIR_AR',
                                         'RLI_TDA_NOIR_MYGA',
                                         'AADE_DD06_TDA_NOIR', 'AEGON_TDA',
                                         'LIBERTY_TDA', 'AHL_LNC_FIA_NOIR',
                                         'AHL_LNC_TDA_NOIR', 'LNC_FIA_P2018',
                                         'AANY_TDA',
                                         'ALICNY_PAYOUT_LC',
                                         'ALICNY_PAYOUT_LC_SC',
                                         'ALICNY_PAYOUT_NLC',
                                         'ALICNY_PAYOUT_NLC_SC',
                                         'ALICNY_TDA'],
                         'company_list': ['DELLIC', 'DEIIC', 'VOYA',
                                          'AANY', 'ALICNY', 'BNY'],
                         'stat_files': (7, 8, 9, 10, 11)},

                'aade': {'calcs': [('payout', 58), ('f133', 4), ('stat', None), ('dac', 1),
                                   ('sop_pre', 2), ('sop_full', 3), ('rm', 12)],
                         'filter_type': 'keep',
                         'cohort_list': ['DA', 'EIA', 'EIX12', 'EIX13',
                                         'EIX14', 'EIX15', 'EIXACQI',
                                         'EIXACQM', 'EIXPE13', 'EIXPE14',
                                         'EIXPE15', 'GWB12', 'GWB13', 'GWB14',
                                         'GWB15', 'GWBACQ', 'GWBBEN12',
                                         'GWBBEN13', 'GWBBEN14', 'GWBBEN15',
                                         'MYGA', 'REG11', 'REG12', 'REG13',
                                         'REG14', 'REG15', 'REGACQ',
                                         'AEGON_TDA', 'LIBERTY_TDA'],
                         'company_list': ['DELLIC', 'DEIIC', 'AADE', 'IIC'],
                         'stat_files': (7, None, None, 10, None)},

                'voya': {'calcs': [('f133', 44), ('payout', 57),
                                   ('stat', None), ('dac', 41),
                                   ('sop_pre', 42), ('sop_full', 43),
                                   ('rm', 12)],
                         'filter_type': 'keep',
                         'cohort_list': ['VIAC_FIA_IR', 'VIAC_FIA_NON',
                                         'VIAC_GMIB_LC_SC', 'VIAC_GMIB_NLC_SC',
                                         'VIAC_PAYOUT_LC', 'VIAC_PAYOUT_LC_SC',
                                         'VIAC_PAYOUT_NLC',
                                         'VIAC_PAYOUT_NLC_SC', 'VIAC_SA',
                                         'VIAC_TDA_NOIR_AR',
                                         'VIAC_TDA_NOIR_MYGA', 'RLI_FIA_IR',
                                         'RLI_FIA_NON', 'RLI_PAYOUT_LC',
                                         'RLI_PAYOUT_LC_SC', 'RLI_PAYOUT_NLC',
                                         'RLI_PAYOUT_NLC_SC', 'RLI_SA',
                                         'RLI_TDA_NOIR_AR',
                                         'RLI_TDA_NOIR_MYGA',
                                         'VIAC_FIA_NON_ACQ',
                                         'VIAC_FIA_IR_ACQ'],
                         'company_list': ['VOYA'],
                         'stat_files': (47, None, None, 50, None)},

                'rocky': {'calcs': [('f133', 44), ('stat', None), ('dac', 41),
                                    ('rm', 12)],
                         'filter_type': 'keep',
                         'cohort_list': ['AHL_LNC_FIA_NOIR',
                                         'AHL_LNC_TDA_NOIR', 'LNC_FIA_P2018'],
                         'company_list': ['LNC'],
                         'stat_files': (47, None, None, 50, None)},


                'dd04': {'calcs': [('f133', 4), ('dac', 1), ('rm', 12)],
                         'filter_type': 'keep',
                         'cohort_list': ['AADE_DD04_FIA_NOIR',
                                         'AADE_DD04_TDA_NOIR',
                                         'DD04_NON_2017', 'DD04_NON_2018',
                                         'DD04_NON_2019'],
                         'company_list': []},

                'dd05': {'calcs': [('dac', 1), ('rm', 12)],
                         'filter_type': 'keep',
                         'cohort_list': ['AADE_DD05_TDA_NOIR'],
                         'company_list': []},

                'dd06': {'calcs': [('dac', 1), ('rm', 12)],
                         'filter_type': 'keep',
                         'cohort_list': ['AADE_DD06_TDA_NOIR'],
                         'company_list': []},

                'alre': {'calcs': [('f133', 142), ('dac', 143),
                                   ('sop_pre', 145), ('sop_full', 144),
                                   ('rm', 141)],
                         'filter_type': None,
                         'cohort_list': [],
                         'company_list': []},

                'aany': {'calcs': [('payout', 17),
                                   ('stat', None), ('dac', 1),

                                   ('rm', 12)],
                         'filter_type': 'keep',
                         'cohort_list': ['AANY_TDA',
                                         'ALICNY_PAYOUT_LC',
                                         'ALICNY_PAYOUT_LC_SC',
                                         'ALICNY_PAYOUT_NLC',
                                         'ALICNY_PAYOUT_NLC_SC',
                                         'ALICNY_TDA'],
                         'company_list': ['AANY', 'ALICNY', 'BNY'],
                         'stat_files': (7, 8, 9, 10, 11)},
                }

    return settings


def lob_settings(lob, settings):
    if 'stat_files' in settings[lob]:
        stat_files = settings[lob]['stat_files']
    calc_dict = {}
    for c, r in settings[lob]['calcs']:
        if c == 'stat':
            calc_dict[c] = []
            for n in stat_files:
                if n:
                    calc_dict[c].append(str(n).zfill(3))
                else:
                    calc_dict[c].append(None)
        else:
            calc_dict[c] = [str(r).zfill(3)]
    filter_type = settings[lob]['filter_type']
    cohort_list = settings[lob]['cohort_list']
    company_list = settings[lob]['company_list']
    return list(calc_dict), calc_dict, filter_type, cohort_list, company_list


def get_calc_var(calc, settings):
    calc_var = settings[calc]['calc_var']
    if not calc_var:
        variables = settings[calc]['variables']
        calc_var = [v.replace('_ts_E0', '') for v in variables]
    return calc_var


def init_inputs_global(calc_settings_dict, lob_settings_dict, models):
    lob_list = list(lob_settings_dict)
    calcs_list = list(calc_settings_dict)
    calc_var_list = []
    for c in calcs_list:
        for v in get_calc_var(c, calc_settings_dict):
            calc_var_list.append(v)
    models.insert(0, 'diff')
    index = pd.MultiIndex.from_product([models, calc_var_list])
    df = pd.DataFrame(index=index, columns=lob_list)
    return df, lob_list


def find_file(folder, ain, run_num, res_file):
    def get_file(pattern, run_num):
        matches = glob(pattern.format(run_num))
        matches = [f for f in matches if '.afd' not in f.lower()]
        if len(matches) < 1:
            return ''
        else:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]

    pattern = os.path.join(folder, f'{ain}.Proj.{{0}}.Run.{{0}}.*{res_file}*')
    if len(run_num) == 1:
        return get_file(pattern, run_num[0])
    else:
        file_dict = {}
        stat_list = ['AG33', 'ss', 'bar', 'tax', 'tax_43']
        for r, s in zip(run_num, stat_list):
            if r:
                file_dict[s] = get_file(pattern, r)
            else:
                file_dict[s] = None
        return file_dict


def get_file_state(results, summary):
    if type(results) == dict:
        check_res = os.path.isfile(results['AG33'])
        file_times = []
        for f in [v for v in results.values() if v]:
            if os.path.isfile(f):
                file_times.append(os.path.getmtime(f))
        res_time = max(file_times) if check_res else 0
    else:
        check_res = os.path.isfile(results)
        res_time = os.path.getmtime(results) if check_res else 0
    check_sum = os.path.isfile(summary)
    sum_time = os.path.getmtime(summary) if check_sum else 0
    if sum_time > res_time:
        return 1    # summary file already exists and is current
    elif check_res:
        return 0    # summary file does not exist or is not current
    else:
        return 2    # model output file does not exist


def get_columns(results_file, variables, filter_column,
                use_var_col, calc_type):
    if filter_column:
        columns = [filter_column]
    else:
        columns = []
    if use_var_col:
        columns.append('VarName')
        with open(results_file) as f:
            all_columns = f.readline().strip().split('\t')
        if 'Cyclical' in results_file:
            val_cols = [c for c in all_columns if '/' in c]
        else:
            val_cols = [c for c in all_columns if 'Value' in c]
        if calc_type == 'nb_res':
            columns.append(val_cols[1])
        elif calc_type:
            columns.extend(val_cols)
        else:
            columns.append(val_cols[0])
    else:
        columns.extend(variables)
    return columns


def DAC_cols(df):
    nier = np.genfromtxt('NIER.txt')
    nier = np.append(np.insert(nier, 0, 0), nier[-1])
    df.loc['Value201'] = 0
    df['AVIF'] = df['AVIF'].rolling(2).mean() * ((1 + nier) ** .25 - 1)
    df['AOptValue'] = df['AOptValue'].diff()
    return df


def rm_cols(df):
    df.loc['Value001':, ['TabRes', 'TotalResCeded']] = 0
    df.loc['Value001':, 'PolLoan'] = df['PolLoan'].diff()
    return df


def rm_calcs(df, sign):
    zero_first_val = [c for c in df if c not in ['TabRes', 'TotalResCeded']]
    df.loc['Value000', zero_first_val] = 0
    df = (df * sign).sum(axis=1)
    duration = np.arange(.25, 50.25, .25)
    wal = np.average(duration, weights=df.loc['Value001':])
    cof = (1 + np.irr(df)) ** 4 - 1
    df = pd.DataFrame({'WAL': [wal], 'COF': [cof]})
    return df


def nb_calc(df, sign):
    first_values = df.loc[['Value001', 'Value002'], 'CashPrem'].values
    df.loc[['Value000', 'Value001'], 'CashPrem'] = first_values
    df = (df * sign).sum(axis=1)
    cof = (1 + np.irr(df)) ** 4 - 1
    return cof


def filter_lob(df, filter_type, column, values):
    if filter_type == 'keep':
        df = df[df[column].isin(values)]
    elif filter_type == 'drop':
        df = df[~df[column].isin(values)]
    df = df.drop(column, axis=1)
    return df


def stat_calcs(file_dict, variables, calc_var,
               filter_type, filter_column, filter_values):

    def read_file(file_name, file_dict, cols, filter_type, column, values):
        path = file_dict[file_name]
        if path is None or len(path) == 0:
            return None
        else:
            print(f'\t\t\treading {file_name}')
            usecols = [column] + cols
            df = pd.read_csv(path, sep='\t', usecols=usecols)
            df = filter_lob(df, filter_type, column, values)
            if 'AdminSystem' in df:
                df = df[df['AdminSystem'] != 'CSC']
                df = df.drop('AdminSystem', axis=1)
            return df

    cv_iapp_col = variables['cv_iapp']
    cv_nopp_col = variables['cv_nopp']
    stat_col = variables['stat']
    tax_col = variables['tax']
    ss_col = variables['ss']
    av = variables['av']
    av_floor_pct = variables['av_floor']
    de_pct = variables['de_pct']
    han_net = variables['han_net']
    wbre = variables['wbre']

    col_dict = {'AG33': [cv_iapp_col, stat_col, av, av_floor_pct, de_pct,
                         'AdminSystem', 'AG33apply'],
                'bar': [cv_iapp_col, stat_col],
                'ss': [ss_col],
                'tax': [cv_nopp_col, stat_col, tax_col, han_net,
                        wbre, 'AdminSystem'],
                'tax_43': [cv_nopp_col, stat_col]}

    df_dict = {}
    
    for calc in file_dict:
        df_dict[calc] = read_file(calc, file_dict, col_dict[calc],
                                  filter_type, filter_column, filter_values)

    AG33 = df_dict['AG33']
    bar = df_dict['bar']
    ss = df_dict['ss']
    tax = df_dict['tax']
    tax_43 = df_dict['tax_43']

    tax_flag = tax is not None
    AG43_flag = bar is not None and ss is not None
    AG43_tax_flag = tax_43 is not None
    
    AG33['av_floor'] = AG33[av] * AG33[av_floor_pct]
    AG33['STATRESERVE'] = AG33[[cv_iapp_col, stat_col, 'av_floor']].max(axis=1)

    if tax_flag:
        tax = tax.rename(columns={cv_nopp_col: 'cv_tax'})

        tax['de_base'] = np.where(tax[wbre] == '_',
                                  tax[stat_col],
                                  tax[han_net])

        tax_columns = ['cv_tax', 'de_base', tax_col]
        AG33[tax_columns] = tax[tax_columns]

        AG33['DE_STATRESERVE'] = (AG33[['cv_tax', 'de_base', 'av_floor']]
                                  .max(axis=1)
                                  * AG33[de_pct])

        de_mask = AG33[de_pct] > .99

        AG33['taxcap'] = np.where(de_mask,
                                  AG33['DE_STATRESERVE'],
                                  AG33['STATRESERVE'])

        AG33['tax_nofloor'] = (np.where(de_mask,
                                       AG33[['DE_STATRESERVE',
                                             'av_floor']].max(axis=1),
                                       AG33[tax_col])
                                       * .9281)

        AG33['TAXRESERVE'] = (AG33[['tax_nofloor', 'cv_tax']]
                              .max(axis=1))

        AG33['TAXRESERVE'] = AG33[['TAXRESERVE', 'taxcap']].min(axis=1)

    for c in calc_var:
        if c in AG33:
            AG33.loc[:, c] = AG33[c] * AG33['AG33apply']

    if AG43_flag:
        bar['nofloor'] = bar[stat_col] + ss[ss_col]
        bar['STATRESERVE'] = bar[['nofloor', cv_iapp_col]].max(axis=1)

        if AG43_tax_flag:
            tax_43['taxbase'] = (tax_43[stat_col] + ss[ss_col]) * .9281
            bar['tax_no_cap'] = tax_43[['taxbase', cv_nopp_col]].max(axis=1)
            bar['TAXRESERVE'] = bar[['STATRESERVE', 'tax_no_cap']].min(axis=1)
            bar['DE_STATRESERVE'] = np.nan

    stat_sum = pd.Series(dict(zip(calc_var, [np.nan] * 3)))
    stat_sum.STATRESERVE = AG33['STATRESERVE'].sum()
    if AG43_flag:
        stat_sum.STATRESERVE += bar['STATRESERVE'].sum()
    if tax_flag:
        stat_sum[calc_var[1:]] = AG33[calc_var[1:]].sum()
    if AG43_tax_flag:
        stat_sum[calc_var[1:]] += bar[calc_var[1:]].sum()

    return pd.DataFrame(stat_sum).T


def do_calcs(results_file, summary_file, variables, sign, calc_type,
             filter_type, filter_column, filter_values, calc_var):

    file_state = get_file_state(results_file, summary_file)
    if file_state == 2:
        return [np.nan] * len(calc_var)
    else:
        if file_state == 1:
            df = pd.read_csv(summary_file, sep='\t')
        else:
            if calc_type == 'stat':
                df = stat_calcs(results_file, variables, calc_var,
                                filter_type, filter_column, filter_values)
            else:
                var_col_files = ['Cyclical', 'SubTotal', 'Total001']

                use_var_col = len([c for c in var_col_files
                                   if c in results_file])

                usecols = get_columns(results_file, variables, filter_column,
                                      use_var_col, calc_type)

                df = pd.read_csv(results_file, sep='\t', usecols=usecols)

                if filter_column:
                    df[filter_column] = df[filter_column].fillna('n/a')

                    df = filter_lob(df, filter_type, filter_column,
                                    filter_values)

                    if min(df.shape) == 0:
                        return [np.nan] * len(calc_var)

                if use_var_col:
                    df = df[df.VarName.isin(variables)]
                    df = df.groupby('VarName')
                df = pd.DataFrame(df.sum()).transpose()
                df = df[variables]
                if calc_type and calc_type != 'nb_res':
                    if calc_type == 'PV_egp':
                        df = DAC_cols(df)
                        rate = 1.025 ** .25 - 1
                    else:
                        rate = .025 / 4
                        if calc_type == 'rm':
                            df = rm_cols(df)
                            df_rm = rm_calcs(df, sign)
                    if calc_type == 'nb_cof':
                        cof = nb_calc(df, sign)
                        df = pd.DataFrame({'cof': [cof]})
                    else:
                        df = df.iloc[1:]

                        pv_ben = (np.npv(rate, (df * sign).sum(axis=1))
                                  / (1 + rate))

                        df = pd.DataFrame({'PV': [pv_ben]})
                        if calc_type == 'rm':
                            df = pd.concat([df, df_rm], axis=1)
            df.to_csv(summary_file, sep='\t', index=False)
        return tuple(df.iloc[0])


def main():
    file_dict, delete_temp_files = get_config()
    base_name, dev_name = list(file_dict)
    calc_settings_dict = calc_settings()
    lob_settings_dict = get_lob_settings_dict()
    df_summary, lob_list = init_inputs_global(calc_settings_dict,
                                              lob_settings_dict,
                                              [base_name, dev_name])

    for model, file_inputs in file_dict.items():
        print(model)
        folder, ain = file_inputs

        temp_folder = os.path.join(folder, '.results_summary_temp')
        if delete_temp_files:
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
            os.makedirs(temp_folder)
        else:
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)

        for lob in lob_list:
            print('\t{}'.format(lob))
            (calcs, calc_dict, filter_type,
             cohort_list, company_list) = lob_settings(lob, lob_settings_dict)
            for c in calcs:
                print('\t\t{}'.format(c))
                variables = calc_settings_dict[c]['variables']
                sign = calc_settings_dict[c]['sign']
                calc_type = calc_settings_dict[c]['calc_type']
                res_file = calc_settings_dict[c]['res_file']
                filter_column = calc_settings_dict[c]['filter_column']
                filter_list_type = calc_settings_dict[c]['filter_list_type']
                if filter_list_type == 'cohort':
                    filter_values = cohort_list
                else:
                    filter_values = company_list

                calc_var = get_calc_var(c, calc_settings_dict)
                run_num = calc_dict[c]
                results_file = find_file(folder, ain, run_num, res_file)
                summary_file = f'results_summary_{lob}_{c}_{model}.txt'
                summary_file = os.path.join(temp_folder, summary_file)

                calc_results = do_calcs(results_file, summary_file,
                                        variables, sign, calc_type,
                                        filter_type, filter_column,
                                        filter_values, calc_var)

                df_summary.loc[(model, calc_var), lob] = calc_results

    diff_ind = df_summary.index.levels[1]
    df_summary.loc['diff'] = (df_summary.loc[dev_name]
                              - df_summary.loc[base_name]).values
    df_summary = df_summary.fillna('n/a')
    df_summary.to_csv('Results_Summary_{}.csv'.format(dev_name))


main()
