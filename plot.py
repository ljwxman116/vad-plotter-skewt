
from __future__ import print_function

import numpy as np

import matplotlib as mpl
mpl.use('agg')
import pylab
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import json
from datetime import datetime, timedelta

from params import vec2comp

_seg_hghts = [0, 3, 6, 9, 12, 18]
_seg_colors = ['r', '#00ff00', '#008800', '#993399', 'c']

def _total_seconds(td):
    return td.days * 24 * 3600 + td.seconds + td.microseconds * 1e-6

def _fmt_timedelta(td):
    seconds = int(_total_seconds(td))
    periods = [
            ('dy', 60*60*24),
            ('hr',    60*60),
            ('min',      60),
            ('sec',       1)
            ]

    strings=[]
    for period_name,period_seconds in periods:
            if seconds > period_seconds:
                    period_value, seconds = divmod(seconds,period_seconds)
                    strings.append("%s %s" % (period_value, period_name))

    return " ".join(strings)


def _plot_param_table(parameters, web=False):
    storm_dir, storm_spd = parameters['storm_motion']
    trans = pylab.gca().transAxes
    line_space = 0.028
    start_x = 1.02
    start_y = 1.04 - line_space #1

    line_y = start_y

    kwargs = {'color':'k', 'fontsize':10, 'clip_on':False, 'transform':trans}

    pylab.text(start_x + 0.175, start_y, "Parameter Table", ha='center', fontweight='bold', **kwargs)

    spacer = Line2D([start_x, start_x + 0.361], [line_y - line_space * 0.48] * 2, color='k', linestyle='-', transform=trans, clip_on=False)
    pylab.gca().add_line(spacer)
    line_y -= line_space * 1.5

    pylab.text(start_x + 0.095, line_y - 0.0025, "BWD (kts)", fontweight='bold', **kwargs)
    if not web:
        pylab.text(start_x + 0.22,  line_y - 0.0025, "SRH (m$^2$s$^{-2}$)", fontweight='bold', **kwargs)
    else:
        # Awful, awful hack for matplotlib without a LaTeX distribution
        pylab.text(start_x + 0.22,  line_y - 0.0025, "SRH (m s  )", fontweight='bold', **kwargs)
        pylab.text(start_x + 0.305,  line_y + 0.009, "2   -2", fontweight='bold', color='k', fontsize=6, clip_on=False, transform=trans)

    line_y -= line_space

    pylab.text(start_x, line_y, "0-1 km", fontweight='bold', **kwargs)
    val = "--" if np.isnan(parameters['shear_mag_1km']) else "%d" % int(parameters['shear_mag_1km'])
    pylab.text(start_x + 0.095, line_y, val, **kwargs)
    val = "--" if np.isnan(parameters['srh_1km']) else "%d" % int(parameters['srh_1km'])
    pylab.text(start_x + 0.22,  line_y, val, **kwargs)

    line_y -= line_space

    pylab.text(start_x, line_y, "0-3 km", fontweight='bold', **kwargs)
    val = "--" if np.isnan(parameters['shear_mag_3km']) else "%d" % int(parameters['shear_mag_3km'])
    pylab.text(start_x + 0.095, line_y, val, **kwargs)
    val = "--" if np.isnan(parameters['srh_3km']) else "%d" % int(parameters['srh_3km'])
    pylab.text(start_x + 0.22,  line_y, val, **kwargs)

    line_y -= line_space

    pylab.text(start_x, line_y, "0-6 km", fontweight='bold', **kwargs)
    val = "--" if np.isnan(parameters['shear_mag_6km']) else "%d" % int(parameters['shear_mag_6km'])
    pylab.text(start_x + 0.095, line_y, val, **kwargs)

    spacer = Line2D([start_x, start_x + 0.361], [line_y - line_space * 0.48] * 2, color='k', linestyle='-', transform=trans, clip_on=False)
    pylab.gca().add_line(spacer)
    line_y -= 1.5 * line_space

    pylab.text(start_x, line_y, "Storm Motion:", fontweight='bold', **kwargs)
    val = "--" if np.isnan(parameters['storm_motion']).any() else "%03d/%02d kts" % (storm_dir, storm_spd)
    pylab.text(start_x + 0.26, line_y + 0.001, val, **kwargs)

    line_y -= line_space

    bl_dir, bl_spd = parameters['bunkers_left']
    pylab.text(start_x, line_y, "Bunkers Left Mover:", fontweight='bold', **kwargs)
    val = "--" if np.isnan(parameters['bunkers_left']).any() else "%03d/%02d kts" % (bl_dir, bl_spd)
    pylab.text(start_x + 0.26, line_y + 0.001, val, **kwargs)

    line_y -= line_space

    br_dir, br_spd = parameters['bunkers_right']
    if not web:
        pylab.text(start_x, line_y, "Bunkers Right Mover:", fontweight='bold', **kwargs)
    else:
        pylab.text(start_x, line_y - 0.005, "Bunkers Right Mover:", fontweight='bold', **kwargs)
    val = "--" if np.isnan(parameters['bunkers_right']).any() else "%03d/%02d kts" % (br_dir, br_spd)
    if not web:
        pylab.text(start_x + 0.26, line_y + 0.001, val, **kwargs)
    else:
        pylab.text(start_x + 0.26, line_y - 0.001, val, **kwargs)

    line_y -= line_space

    mn_dir, mn_spd = parameters['mean_wind']
    pylab.text(start_x, line_y, "0-6 km Mean Wind:", fontweight='bold', **kwargs)
    val = "--" if np.isnan(parameters['mean_wind']).any() else "%03d/%02d kts" % (mn_dir, mn_spd)
    pylab.text(start_x + 0.26, line_y + 0.001, val, **kwargs)

    spacer = Line2D([start_x, start_x + 0.361], [line_y - line_space * 0.48] * 2, color='k', linestyle='-', transform=trans, clip_on=False)
    pylab.gca().add_line(spacer)
    line_y -= 1.5 * line_space

    if not web:
        pylab.text(start_x, line_y, "Critical Angle:", fontweight='bold', **kwargs)
        val = "--" if np.isnan(parameters['critical']) else "%d$^{\circ}$" % int(parameters['critical'])
        pylab.text(start_x + 0.18, line_y - 0.00, val, **kwargs)
    else:
        pylab.text(start_x, line_y - 0.0075, "Critical Angle:", fontweight='bold', **kwargs)
        val = "--" if np.isnan(parameters['critical']) else "%d deg" % int(parameters['critical'])
        pylab.text(start_x + 0.18, line_y - 0.0075, val, **kwargs)


def _plot_data(data, parameters):
    storm_dir, storm_spd = parameters['storm_motion']
    bl_dir, bl_spd = parameters['bunkers_left']
    br_dir, br_spd = parameters['bunkers_right']
    mn_dir, mn_spd = parameters['mean_wind']

    u, v = vec2comp(data['wind_dir'], data['wind_spd'])
    alt = data['altitude']

    storm_u, storm_v = vec2comp(storm_dir, storm_spd)
    bl_u, bl_v = vec2comp(bl_dir, bl_spd)
    br_u, br_v = vec2comp(br_dir, br_spd)
    mn_u, mn_v = vec2comp(mn_dir, mn_spd)

    seg_idxs = np.searchsorted(alt, _seg_hghts)
    try:
        seg_u = np.interp(_seg_hghts, alt, u, left=np.nan, right=np.nan)
        seg_v = np.interp(_seg_hghts, alt, v, left=np.nan, right=np.nan)
        ca_u = np.interp(0.5, alt, u, left=np.nan, right=np.nan)
        ca_v = np.interp(0.5, alt, v, left=np.nan, right=np.nan)
    except ValueError:
        seg_u = np.nan * np.array(_seg_hghts)
        seg_v = np.nan * np.array(_seg_hghts)
        ca_u = np.nan
        ca_v = np.nan

    mkr_z = np.arange(16)
    mkr_u = np.interp(mkr_z, alt, u, left=np.nan, right=np.nan)
    mkr_v = np.interp(mkr_z, alt, v, left=np.nan, right=np.nan)

    for idx in range(len(_seg_hghts) - 1):
        idx_start = seg_idxs[idx]
        idx_end = seg_idxs[idx + 1]

        if not np.isnan(seg_u[idx]):
            pylab.plot([seg_u[idx], u[idx_start]], [seg_v[idx], v[idx_start]], '-', color=_seg_colors[idx], linewidth=1.5)

        if idx_start < len(data['rms_error']) and data['rms_error'][idx_start] == 0.:
            # The first segment is to the surface wind, draw it in a dashed line
            pylab.plot(u[idx_start:(idx_start + 2)], v[idx_start:(idx_start + 2)], '--', color=_seg_colors[idx], linewidth=1.5)
            pylab.plot(u[(idx_start + 1):idx_end], v[(idx_start + 1):idx_end], '-', color=_seg_colors[idx], linewidth=1.5)
        else:
            pylab.plot(u[idx_start:idx_end], v[idx_start:idx_end], '-', color=_seg_colors[idx], linewidth=1.5)

        if not np.isnan(seg_u[idx + 1]):
            pylab.plot([u[idx_end - 1], seg_u[idx + 1]], [v[idx_end - 1], seg_v[idx + 1]], '-', color=_seg_colors[idx], linewidth=1.5)

        for upt, vpt, rms in list(zip(u, v, data['rms_error']))[idx_start:idx_end]:
            rad = np.sqrt(2) * rms
            circ = Circle((upt, vpt), rad, color=_seg_colors[idx], alpha=0.05)
            pylab.gca().add_patch(circ)

    pylab.plot(mkr_u, mkr_v, 'ko', ms=10)
    for um, vm, zm in zip(mkr_u, mkr_v, mkr_z):
        if not np.isnan(um):
            pylab.text(um, vm - 0.1, str(zm), va='center', ha='center', color='white', size=6.5, fontweight='bold')

    try:
        pylab.plot([storm_u, u[0]], [storm_v, v[0]], 'c-', linewidth=0.75)
        pylab.plot([u[0], ca_u], [v[0], ca_v], 'm-', linewidth=0.75)
    except IndexError:
        pass

    if not (np.isnan(bl_u) or np.isnan(bl_v)):
        pylab.plot(bl_u, bl_v, 'ko', markersize=5, mfc='none')
        pylab.text(bl_u + 0.5, bl_v - 0.5, "LM", ha='left', va='top', color='k', fontsize=10)

    if not (np.isnan(br_u) or np.isnan(br_v)):
        pylab.plot(br_u, br_v, 'ko', markersize=5, mfc='none')
        pylab.text(br_u + 0.5, br_v - 0.5, "RM", ha='left', va='top', color='k', fontsize=10)

    if not (np.isnan(mn_u) or np.isnan(mn_v)):
        pylab.plot(mn_u, mn_v, 's', color='#a04000', markersize=5, mfc='none')
        pylab.text(mn_u + 0.6, mn_v - 0.6, "MEAN", ha='left', va='top', color='#a04000', fontsize=10)

    smv_is_brm = (storm_u == br_u and storm_v == br_v)
    smv_is_blm = (storm_u == bl_u and storm_v == bl_v)
    smv_is_mnw = (storm_u == mn_u and storm_v == mn_v)

    if not (np.isnan(storm_u) or np.isnan(storm_v)) and not (smv_is_brm or smv_is_blm or smv_is_mnw):
        pylab.plot(storm_u, storm_v, 'k+', markersize=6)
        pylab.text(storm_u + 0.5, storm_v - 0.5, "SM", ha='left', va='top', color='k', fontsize=10)


def _plot_background(min_u, max_u, min_v, max_v):
    max_ring = int(np.ceil(max(
        np.hypot(min_u, min_v),
        np.hypot(min_u, max_v),
        np.hypot(max_u, min_v),
        np.hypot(max_u, max_v)
    )))

    pylab.axvline(x=0, linestyle='-', color='#999999')
    pylab.axhline(y=0, linestyle='-', color='#999999')

    for irng in range(10, max_ring, 10):
        ring = Circle((0., 0.), irng, linestyle='dashed', fc='none', ec='#999999')
        pylab.gca().add_patch(ring)

        if irng <= max_u - 10:
            rng_str = "%d kts" % irng if max_u - 20 < irng <= max_u - 10 else "%d" % irng

            pylab.text(irng + 0.5, -0.5, rng_str, ha='left', va='top', fontsize=9, color='#999999', clip_on=True, clip_box=pylab.gca().get_clip_box())


def plot_hodograph(data, parameters, fname=None, web=False, fixed=False, archive=False):
	
    radll={'TJBQ': ('18.485', '-67.143'),
 'KGLD': ('39.36694', '-101.7'),
 'KBIS': ('46.77083', '-100.76028'),
 'KBOX': ('41.95583', '-71.1375'),
 'KBRO': ('25.91556', '-97.41861'),
 'KDAX': ('38.50111', '-121.67667'),
 'KLBB': ('33.65417', '-101.81361'),
 'KLWX': ('38.97628', '-77.48751'),
 'KMOB': ('30.67944', '-88.23972'),
 'KTFX': ('47.45972', '-111.38444'),
 'KMPX': ('44.84889', '-93.56528'),
 'KRLX': ('38.31194', '-81.72389'),
 'KTLH': ('30.3975', '-84.32889'),
 'KYUX': ('32.49528', '-114.65583'),
 'PAEC': ('64.51139', '-165.295'),
 'RKJK': ('35.92417', '126.62222'),
 'KPBZ': ('40.53167', '-80.21833'),
 'KAMX': ('25.61056', '-80.41306'),
 'KBYX': ('24.59694', '-81.70333'),
 'KDDC': ('37.76083', '-99.96833'),
 'KFCX': ('37.02417', '-80.27417'),
 'KFTG': ('39.78667', '-104.54528'),
 'KICX': ('37.59083', '-112.86222'),
 'KLOT': ('41.60444', '-88.08472'),
 'KLVX': ('37.97528', '-85.94389'),
 'KLZK': ('34.83639', '-92.26194'),
 'KMHX': ('34.77583', '-76.87639'),
 'KMLB': ('28.11306', '-80.65444'),
 'KOTX': ('47.68056', '-117.62583'),
 'PACG': ('56.85278', '-135.52917'),
 'KHDX': ('33.07639', '-106.12222'),
 'KTYX': ('43.75583', '-75.68'),
 'PHWA': ('19.095', '-155.56889'),
 'PAPD': ('65.03556', '-147.49917'),
 'PGUA': ('13.45444', '144.80833'),
 'PHKI': ('21.89417', '-159.55222'),
 'RKSG': ('36.95972', '127.01833'),
 'KCBW': ('46.03917', '-67.80694'),
 'KBBX': ('39.49611', '-121.63167'),
 'KBUF': ('42.94861', '-78.73694'),
 'KGGW': ('48.20639', '-106.62417'),
 'KGRK': ('30.72167', '-97.38278'),
 'KJAX': ('30.48444', '-81.70194'),
 'KILX': ('40.15056', '-89.33667'),
 'KIWX': ('41.40861', '-85.7'),
 'KLTX': ('33.98917', '-78.42917'),
 'KMQT': ('46.53111', '-87.54833'),
 'KMSX': ('47.04111', '-113.98611'),
 'KMXX': ('32.53667', '-85.78972'),
 'KNQA': ('35.34472', '-89.87333'),
 'KOHX': ('36.24722', '-86.5625'),
 'KPAH': ('37.06833', '-88.77194'),
 'KPUX': ('38.45944', '-104.18139'),
 'KUEX': ('40.32083', '-98.44167'),
 'LPLA': ('38.73028', '-27.32167'),
 'PHKM': ('20.12556', '-155.77778'),
 'TJRV': ('18.256', '-65.637'),
 'KARX': ('43.82278', '-91.19111'),
 'KHGX': ('29.47194', '-95.07889'),
 'KBHX': ('40.49833', '-124.29194'),
 'KGYX': ('43.89139', '-70.25694'),
 'KHPX': ('36.73667', '-87.285'),
 'KLSX': ('38.69889', '-90.68278'),
 'KMUX': ('37.15528', '-121.8975'),
 'KNKX': ('32.91889', '-117.04194'),
 'KABX': ('35.14972', '-106.82333'),
 'KAKQ': ('36.98389', '-77.0075'),
 'KAPX': ('44.90722', '-84.71972'),
 'KBGM': ('42.19972', '-75.985'),
 'KDIX': ('39.94694', '-74.41111'),
 'KDYX': ('32.53833', '-99.25417'),
 'KESX': ('35.70111', '-114.89139'),
 'KFDR': ('34.36222', '-98.97611'),
 'KFFC': ('33.36333', '-84.56583'),
 'KGSP': ('34.88306', '-82.22028'),
 'KMBX': ('48.3925', '-100.86444'),
 'KHNX': ('36.31417', '-119.63111'),
 'KHTX': ('34.93056', '-86.08361'),
 'KIND': ('39.7075', '-86.28028'),
 'KRIW': ('43.06611', '-108.47667'),
 'KSFX': ('43.10583', '-112.68528'),
 'KSGF': ('37.23528', '-93.40028'),
 'KVAX': ('30.89', '-83.00194'),
 'KEMX': ('31.89361', '-110.63028'),
 'KFWS': ('32.57278', '-97.30278'),
 'KINX': ('36.175', '-95.56444'),
 'KMRX': ('36.16833', '-83.40194'),
 'KOAX': ('41.32028', '-96.36639'),
 'KSOX': ('33.81778', '-117.635'),
 'KVWX': ('38.26', '-87.7247'),
 'KGWX': ('33.89667', '-88.32889'),
 'KMAX': ('42.08111', '-122.71611'),
 'KOKX': ('40.86556', '-72.86444'),
 'KRGX': ('39.75417', '-119.46111'),
 'KVBX': ('34.83806', '-120.39583'),
 'PAKC': ('58.67944', '-156.62944'),
 'KCLE': ('41.41306', '-81.86'),
 'KEOX': ('31.46028', '-85.45944'),
 'KMAF': ('31.94333', '-102.18889'),
 'KVNX': ('36.74083', '-98.1275'),
 'KLGX': ('47.1158', '-124.1069'),
 'KPDT': ('45.69056', '-118.85278'),
 'KVTX': ('34.41167', '-119.17861'),
 'KDFX': ('29.2725', '-100.28028'),
 'KAMA': ('35.23333', '-101.70889'),
 'KATX': ('48.19472', '-122.49444'),
 'KBMX': ('33.17194', '-86.76972'),
 'KDGX': ('32.28', '-89.98444'),
 'KDLH': ('46.83694', '-92.20972'),
 'KDOX': ('38.82556', '-75.44'),
 'KDTX': ('42.69972', '-83.47167'),
 'KEAX': ('38.81028', '-94.26417'),
 'KENX': ('42.58639', '-74.06444'),
 'KFDX': ('34.63528', '-103.62944'),
 'KFSX': ('34.57444', '-111.19833'),
 'KGJX': ('39.06222', '-108.21306'),
 'KGRR': ('42.89389', '-85.54472'),
 'KMKX': ('42.96778', '-88.55056'),
 'KICT': ('37.65444', '-97.4425'),
 'KILN': ('39.42028', '-83.82167'),
 'KLIX': ('30.33667', '-89.82528'),
 'KLRX': ('40.73972', '-116.80278'),
 'KMTX': ('41.26278', '-112.44694'),
 'KSHV': ('32.45056', '-93.84111'),
 'KTBW': ('27.70528', '-82.40194'),
 'KTLX': ('35.33306', '-97.2775'),
 'KTWX': ('38.99694', '-96.2325'),
 'KCAE': ('33.94861', '-81.11861'),
 'KBLX': ('45.85389', '-108.60611'),
 'KCBX': ('43.49083', '-116.23444'),
 'KCXX': ('44.51111', '-73.16639'),
 'KJKL': ('37.59083', '-83.31306'),
 'KEWX': ('29.70361', '-98.02806'),
 'KGRB': ('44.49833', '-88.11111'),
 'KJGX': ('32.675', '-83.35111'),
 'KPOE': ('31.15528', '-92.97583'),
 'KRAX': ('35.66528', '-78.49'),
 'KRTX': ('45.715', '-122.96417'),
 'KSJT': ('31.37111', '-100.49222'),
 'KUDX': ('44.125', '-102.82944'),
 'PABC': ('60.79278', '-161.87417'),
 'KEYX': ('35.09778', '-117.56'),
 'KFSD': ('43.58778', '-96.72889'),
 'KSRX': ('35.29056', '-94.36167'),
 'PHMO': ('21.13278', '-157.18'),
 'RODN': ('26.30194', '127.90972'),
 'KCYS': ('41.15194', '-104.80611'),
 'KABR': ('45.45583', '-98.41306'),
 'KCLX': ('32.65556', '-81.04222'),
 'KCRP': ('27.78389', '-97.51083'),
 'KDVN': ('41.61167', '-90.58083'),
 'KEPZ': ('31.87306', '-106.6975'),
 'KEVX': ('30.56417', '-85.92139'),
 'KIWA': ('33.28917', '-111.66917'),
 'KLCH': ('30.125', '-93.21583'),
 'KLNX': ('41.95778', '-100.57583'),
 'KMVX': ('47.52806', '-97.325'),
 'PAHG': ('60.72591', '-151.35146'),
 'PAIH': ('59.46194', '-146.30111'),
 'TJUA': ('18.1175', '-66.07861'),
 'KDMX': ('41.73111', '-93.72278'),
 'KCCX': ('40.92306', '-78.00389')}
	
    img_title = "%s VWP %s" % (data.rid, data['time'].strftime("%m-%d-%Y %H:%M:%SZ"))
    if fname is not None:
        img_file_name = fname
    else:
        img_file_name = "%s_vad.png" % data.rid

    u, v = vec2comp(data['wind_dir'], data['wind_spd'])

    sat_age = 6 * 3600
    if fixed or len(u) == 0:
        ctr_u, ctr_v = 20, 20
        size = 120
    else:
        ctr_u = u.mean()
        ctr_v = v.mean()
        size = max(u.max() - u.min(), v.max() - v.min()) + 20
        size = max(120, size)

    min_u = ctr_u - size / 2
    max_u = ctr_u + size / 2
    min_v = ctr_v - size / 2
    max_v = ctr_v + size / 2

    now = datetime.utcnow()
    img_age = now - data['time']
    age_cstop = min(_total_seconds(img_age) / sat_age, 1) * 0.4
    age_color = mpl.cm.get_cmap('hot')(age_cstop)[:-1]

    age_str = "Image Generation: %s [%s old]" % (now.strftime("%m-%d-%Y %H:%M:%SZ"), _fmt_timedelta(img_age))

    pylab.figure(figsize=(10, 7.5), dpi=300) #7.5
    fig_wid, fig_hght = pylab.gcf().get_size_inches()
    fig_aspect = fig_wid / fig_hght

    axes_left = 0.005 #0.05 
    axes_bot = 0.03 #0.05
    axes_hght = 0.93 #0.9
    axes_wid = axes_hght / fig_aspect
    pylab.axes((axes_left, axes_bot, axes_wid, axes_hght))

    _plot_background(min_u, max_u, min_v, max_v)
    _plot_data(data, parameters)
    _plot_param_table(parameters, web=web)

    pylab.xlim(min_u, max_u)
    pylab.ylim(min_v, max_v)
    pylab.xticks([])
    pylab.yticks([])

    pylab.title(img_title, color=age_color,fontweight='bold')
    pylab.text(0., -0.01, age_str, transform=pylab.gca().transAxes, ha='left', va='top', fontsize=9, color=age_color)
    web_brand = "http://www.autumnsky.us/vad/"
    pylab.text(1.0, -0.01, web_brand, transform=pylab.gca().transAxes, ha='right', va='top', fontsize=9)
  
    ### Skew-T Plot ###
    from matplotlib.ticker import (MultipleLocator, NullFormatter,
                                   ScalarFormatter)
    from matplotlib.projections import register_projection
    from skew import SkewXAxes
    from ftplib import FTP
    import subprocess
    import re
    import time
    import metpy.calc as mpcalc
    from metpy.units import units
    
    ftp = FTP('ftp.ncep.noaa.gov')
    ftp.login()
    ftp.cwd('pub/data/nccf/com/rap/prod/')
    drl=ftp.nlst()
    dr=[]
    for drn in drl:
        if 'rap.' in drn:
            dr.append(drn)
    ftp.cwd(dr[-1])
    drld=ftp.nlst()
    fn=[]
    for nm in drld:
        if '.awp130pgrbf' in nm and '.idx' not in nm:
            fn.append(nm)
    urlb='https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/%s/'%dr[-1]
    urlt={}
    for fnm in drld:
        if '.awp130pgrbf' in fnm:
            if '.idx' not in fnm:
                init=re.findall('t[0-9]{2}z',fn[-1],re.DOTALL)
                init=''.join(init).strip('t')
                if init in fnm:
                    if 'f00' not in fnm:
                        fh=re.findall('f[0-9]{2}.',fnm,re.DOTALL)
                        fh=int(''.join(fh).strip('f').strip('.').lstrip('0'))
                        urlt[fh]=fnm
    if len(urlt)==0:
        print('trying %s'%dr[-2])
        ftp.cwd('/pub/data/nccf/com/rap/prod/%s'%dr[-2])
        drld=ftp.nlst()
        fn=[]
        for nm in drld:
            if '.awp130pgrbf' in nm and '.idx' not in nm:
                fn.append(nm)
        urlb='https://www.ftp.ncep.noaa.gov/data/nccf/com/rap/prod/%s/'%dr[-2]
        urlt={}
        for fnm in drld:
            if '.awp130pgrbf' in fnm:
                if '.idx' not in fnm:
                    init=re.findall('t[0-9]{2}z',fn[-1],re.DOTALL)
                    init=''.join(init).strip('t')
                    if init in fnm:
                        if 'f00' not in fnm:
                            fh=re.findall('f[0-9]{2}.',fnm,re.DOTALL)
                            fh=int(''.join(fh).strip('f').strip('.').lstrip('0'))
                            urlt[fh]=fnm
    urll=list(urlt.items())[0][1]
    url=urlb+urll
    
    latr=float(radll[data.rid][0]) #ns
    lonr=float(radll[data.rid][1]) #we
    
    fp='' #file path directory for data download
    f='' #file pat directory for 'get_inv.pl' and 'get_grib.pl'
    ic=subprocess.Popen(f'{f}./get_inv.pl {url}.idx | egrep "(:RH:|:TMP:|:PRES:surface:|:CAPE:surface:)" | {f}./get_grib.pl {url} {fp}rap.grib2',shell=True)
    ic.wait()
    of=subprocess.Popen(f'/usr/local/bin/wgrib2 {fp}rap.grib2 -s -lon {lonr} {latr} > {fp}rap.txt', shell=True)
    of.wait()
    itc=(subprocess.Popen(f'/usr/local/bin/wgrib2 {fp}rap.grib2 -end_ft | tail -1 | /usr/bin/egrep -E -o "[0-9]{{10}}"',shell=True,stdout=subprocess.PIPE)).stdout.read().decode('utf-8')
    it=itc.strip('\n') 
    its=time.strptime(it,'%Y%m%d%H')
    vt=time.strftime("%m-%d-%Y %H:%MZ",its)
    
    dat=open(f'{fp}rap.txt','r').read()
    
    cpr=re.findall('CAPE:surface(.*?)\n',dat)
    for i in cpr:
        cps=float(''.join(re.findall('val=\d{0,7}',i)).strip('val='))
    
    #T
    tmb=re.findall(':TMP:(.*?)\n',dat)
    tp=[]
    for i in tmb:
        tmbt=re.findall('\d{3,4} mb:',i)
        if len(tmbt)==0:
            pass
        else:
            tp.append(float(''.join([s.strip(' mb:') for s in tmbt])))
    ttl=[]
    for i in tmb:
        tt=re.findall('val=\d{0,5}.\d{0,7}',i)
        ttl.append(''.join([s.strip('val=') for s in tt]))
    t=[]
    for i in range(0,len(tp)):
        #print(f'{rhmbl[i]}: {rhtl[i]}')
        t.append(float(ttl[i])-273.15)
        
    #RH
    rhmb=re.findall(':RH:(.*?)\n',dat)
    rhp=[]
    for i in rhmb:
        rhmbt=re.findall('\d{3,4} mb:',i)
        if len(rhmbt)==0:
            pass
        else:
            rhp.append(float(''.join([s.strip(' mb:') for s in rhmbt])))
    rhtl=[]
    for i in rhmb:
        rht=re.findall('val=\d{0,5}.\d{0,7}',i)
        rhtl.append(''.join([s.strip('val=') for s in rht]))
    rh=[]
    for i in range(0,len(rhp)):
        #print(f'{rhmbl[i]}: {rhtl[i]}')
        rh.append(float(rhtl[i]))
    import math
    def Tdd(T,RH):
        tdd=(243.12*(math.log(RH/100)+((17.62*T)/(243.12+T))))\
    /(17.62-(math.log(RH/100)+((17.62*T)/(243.12+T))))
        return tdd
    td=[]
    for i in range(0,len(t)):
        td.append(Tdd(t[i],rh[i]))
    
    prr=re.findall('PRES:surface(.*?)\n',dat)
    for i in prr:
        ps=float(''.join(re.findall('val=(\d{6}|\d{5}.{1})',i)).strip('val='))/100
    tr=re.findall('TMP:2 m above ground(.*?)\n',dat)
    for i in tr:
        ts=float(''.join(re.findall('val=\d{3}.\d{1,7}',i)).strip('val='))-273.15
    rhr=re.findall('RH:2 m above ground(.*?)\n',dat)
    for i in rhr:
        rhs=float(''.join(re.findall('val=(\d{1,3}.\d{1,7}|\d{0,3})',i)).strip('val='))
        tds=Tdd(ts,rhs)
    
    tpn=[]
    tn=[]
    tdn=[]
    for i in tp:
        if i<ps:
            tpn.append(i)
    for i in range(0,len(tpn)):
        tn.append(t[i])
    for i in range(0,len(tpn)):
        tdn.append(td[i])
    tpn.append(ps)
    tn.append(ts)
    tdn.append(tds)
    
    register_projection(SkewXAxes)

    ax=pylab.axes([.7025, .03, .281, .66],projection='skewx') #left,bottom,width,height
    pylab.grid(True)

    ax.semilogy(tn,tpn,color='C3')
    ax.semilogy(tdn,tpn,color='C2')
    
    if cps>0:
        for i in np.arange(0,65,5):
            prof=mpcalc.moist_lapse(list(reversed(tpn))*units.hPa,i*units.degC,1000*units.hPa).to('degC')
            ax.plot(prof,list(reversed(tpn)),color='k',linestyle=(0, (5, 10)),linewidth=0.5)
        prof=mpcalc.parcel_profile(list(reversed(tpn))*units.hPa,tn[-1]*units.degC,tdn[-1]*units.degC).to('degC')
        ax.plot(prof,list(reversed(tpn)),color='orange',linewidth=0.7)
    
    ax.axvline(0, color='C0',linewidth=.7)

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(np.linspace(100, 1000, 10))
    ax.set_ylim(1050, 100)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis='both',which='both',labelsize=4,pad=1,length=0)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlim(-50, 50)
    ax.text(0.58, -0.03,f'RAP 01HR: {vt}',transform=ax.transAxes,fontweight='bold',fontsize=5)
    ax.xaxis.label.set_visible(False)
    
    pylab.savefig(img_file_name, dpi=300) #dpi=pylab.gcf().dpi)
    pylab.close()