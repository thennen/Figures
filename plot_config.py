# Import this in all the figure generating scripts in order to share parameters
# So I can split the files up as I see fit but still be able to easily find the file that generated every figure
# There's also a function that can find all the figure generating scripts and try to run them all

# Use these lines at the top of each figure generating script
'''
import plot_config
from importlib import reload
reload(plot_config)
from plot_config import *
'''

import inspect
import os
import re
import matplotlib as mpl
import numpy as np
import PIL
from matplotlib import pylab as plt
from matplotlib.widgets import AxesWidget
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from functools import wraps
import pandas as pd
from cycler import cycler
import importlib
import socket
import fnmatch
from copy import deepcopy

pjoin = os.path.join

# Where's the data relative to the host machine?
hostname = socket.gethostname()
# too many terabytes to relocate the data, should all stay in the structure where it was created
if hostname == 'Snufferbatch':
    box = False # box is dead 2023-04
    if box:
        datadir = r'C:\Users\t\Box\RWTH-TH\ivdata'# should be ivdatadir
        comsoldatadir = r'C:\Users\t\Box\RWTH-TH\COMSOL'
        stochasticdatadir = r'C:\Users\t\Box\ReRAM_Stochastic_Shared'
        uXASdatadir = r'C:\Users\t\Box\uXAS'
    else:
        # Mount external drive on H, use box backup
        datadir = r'H:\Box\RWTH-TH\ivdata'
        comsoldatadir = r'H:\Box\RWTH-TH\COMSOL'
        stochasticdatadir = r'H:\Box\ReRAM_Stochastic_Shared'
        uXASdatadir = r'H:\Box\uXAS'

    simdatadir = r'C:\Users\t\sciebo\simulations'
else:
    datadir = simdatadir = comsoldatadir = stochasticdatadir = '.'

# for storing piddley data files from strange sources
localdatadir = r'.\data'

# Specify the horizontal space we have to fit a figure inside
# For this, you have to check latex output of \showthe\columnwidth
inches_per_texpt = 1 / 72.27
# columnwidth = 6.4
# "Doctoral Thesis" template by Steve Gunn and Sunil Patel
columnwidth_pts = 412#.56499
# Revtex (2 column)
# columnwidth = 246 * inches_per_texpt # = 3.404
# but aip says max 3.37 inch
# columnwidth = 3.37
#columnwidth_pts = 504 # full page, Frontiers template
columnwidth = columnwidth_pts * inches_per_texpt

# Aspect ratios
default_aspect = 4/3 # e.g. 640x480
sqrt2 = np.sqrt(2) # like the paper it will be printed on
φ = (1 + np.sqrt(5)) / 2 # golden

# Apply RC settings from the file
# then plots should not depend on what pc or profile you are using
# and should be immune to future changes to default rc settings (looking at you, tick.direction...)
plt.rcParams.update(mpl.rc_params_from_file('matplotlibrc'))

# \showthe\font prints \OT1/ptm/m/n/12 on Frontiers template
# classic (OT1) font encoding, ptm is apparently times new roman
# medium weight (m)
# normal shape (n)
# 12 pt (template has huge font, actual article has much smaller)

# Where your .tex documents are
latex_dir = r'.\tex'

# only if the rc setting text.usetex is True
latex_preamble = r"""
\usepackage{bm}
""".strip()

# Places to choose colors:
# https://medialab.github.io/iwanthue/
# https://coolors.co/generate
# https://projects.susielu.com/viz-palette
# https://colorbrewer2.org
# https://sashamaps.net/docs/resources/20-colors/
# there are more

def rgb2hex(r,g,b):
    if type(r) is float:
        r = int(r*255)
        g = int(g*255)
        b = int(b*255)
    return "#{:02x}{:02x}{:02x}".format(r,g,b)
def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

import seaborn # since 2017 it doesn't mess with the matplotlib style on import
tab10muted = [rgb2hex(*x) for x in seaborn.color_palette('muted')]
tab10deep = [rgb2hex(*x) for x in seaborn.color_palette('deep')]
seaborn_colorblind = [rgb2hex(*x) for x in seaborn.color_palette('colorblind')]


# Shared colors
# HRS and LRS should contrast well, UR and US should contrast well
#paramcolors = ['C0', 'C2', 'C1', 'C3'] 
paramcolors = [tab10deep[i] for i in (0,2,1,3)]


# These are all the different rc settings that I want to use for this publication
rcUpdate = {'interactive'          : True,
            #'font.family'          : 'serif',
            'font.family'          : 'sans-serif', # Apparently it is standard to do figures in sans even when the text is in serif.
            'font.size'            : 10.0,
            'font.serif'           : 'Times New Roman',
            'font.sans-serif'      : ['Helvetica', 'Arial'],
            'legend.fontsize'      : 'small',
            'legend.title_fontsize': 'small',
            'xtick.top'            : True,
            'xtick.direction'      : 'in',
            'ytick.right'          : True,
            'ytick.direction'      : 'in',
            'grid.color'           : 'k',
            'grid.linestyle'       : ':',
            'grid.linewidth'       : 0.5,
            'grid.alpha'           : 0.5,
            #'axes.prop_cycle'      : cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']), # default Tableau (tab10)
            'axes.prop_cycle'      : cycler('color', tab10deep),
            'figure.dpi'           : 200, # Monitor DPI is 96, savefig is 300, but > 200 doesn't always fit on screen
            'figure.figsize'       : [columnwidth, columnwidth*3/4],
            # needs a lot of stuff in the path, massively slow, only needed for \bm,
            # and to make the symbols match tex doc perfectly
            # pukes if you use unicode and for many other potential reasons (it's latex after all)
            'text.usetex'          : False,
            'text.latex.preamble'  : latex_preamble, # could put the journal style file here
            # mathtext is the alternative to usetex, won't apply if usetex=True
            'mathtext.default'     : 'it',
            'mathtext.fontset'     : 'dejavusans',
            #'mathtext.fontset'     : 'cm', # cm uses serif like latex default 'dejavusans' (so symbols look like the text)
            'figure.constrained_layout.use' : True,
            'figure.max_open_warning' : 500,}


plt.rcParams.update(rcUpdate)

# for saving pngs - can look slightly different than the plt.show version if dpi is different
dpi = 300

def valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def savefig(name, fig=None, subdir='images', vector=False, metastring=None, alt=False, allow_interactive=False, **kwargs):
    '''
    Always save figures with filenames starting with the script that generated them
    '''

    # This is if you want to save a figure that probably won't appear in the document
    # but might be useful and you don't want to clutter the main directory
    if alt:
        subdir = os.path.join(subdir, 'alts')

    os.makedirs(subdir, exist_ok=True)

    scriptpath = inspect.stack()[1][1]
    scriptfn = os.path.split(scriptpath)[-1]
    scriptname = os.path.splitext(scriptfn)[0]
    # in case its from interactive or something..
    scriptname = valid_filename(scriptname)

    # somehow this changed between qtconsole versions
    if ('ipython-input' in scriptname or scriptname[:10].isdigit()) and (not allow_interactive):
        print('Not writing figure from interactive console')
        return

    if fig is None:
        fig = plt.gcf()

    savefigkwargs = dict(dpi=dpi)
    savefigkwargs.update(**kwargs)

    # Could go so far as to copy the generating script into the png metadata
    # but won't for now

    metadata = {'description':metastring}
    # this dict is recoverable with PIL.Image.open(filename).info

    pngfn = '_'.join((scriptname, name)) + '.png'
    pngfp = pjoin(subdir, pngfn)
    print(f'Writing {pngfp}')
    fig.savefig(pngfp, metadata=metadata, **savefigkwargs)

    # This was to make vector images/transparent versions one time
    make_extra_formats = False
    if make_extra_formats:
        print('Writing a transparent png version')
        transparentpng_dir = pjoin(subdir, 'transparent_pngs')
        os.makedirs(transparentpng_dir, exist_ok=True)
        transparentpng_fp = pjoin(transparentpng_dir, pngfn)
        fig.savefig(transparentpng_fp, metadata=metadata, transparent=True, **savefigkwargs)
        print('Writing a vector version (pdf)')
        vector_dir = pjoin(subdir, 'vector')
        os.makedirs(vector_dir, exist_ok=True)
        pdffn = '_'.join((scriptname, name)) + '.eps'
        vector_fp = pjoin(vector_dir, pdffn)
        fig.savefig(vector_fp, **savefigkwargs)


    if vector:
        # Stay away from vector for plots with many elements

        # vector format that plays OK with latex and terribly with everything else
        # no transparency, outdated, horrible format
        #plt.savefig('_'.join((scriptname, name)) + '.eps', **savefigkwargs)

        # vector format that no program can open, and latex can't use
        # but you can use it for pasting into ppt
        #plt.savefig('_'.join((scriptname, name)) + '.svg', **savefigkwargs)

        # The one that seems to actually work for most things
        pdffn = '_'.join((scriptname, name)) + '.pdf'
        print(f'Writing {pdffn}')
        fig.savefig(pjoin(subdir, pdffn), **savefigkwargs)

### Standard figure sizes

def widefig(aspect=2, *args, **kwargs):
    ''' As wide as fits on the page '''
    width = columnwidth
    fig, ax = plt.subplots(*args, figsize=(width, width/aspect), **kwargs)
    return fig, ax

def tallfig(aspect=1.3, *args, **kwargs):
    ''' For ncols = 1, potentially stacked subplots
        e.g. tallfig(nrows=2, sharex=True)
    '''
    width = columnwidth * .8
    fig, ax = plt.subplots(*args, figsize=(width, width/aspect), **kwargs)
    return fig, ax

def fig1(aspect=sqrt2, *args, **kwargs):
    '''
    Make a good size fig for a single subplot
    should have significant detail or need space for a color bar
    '''
    width = columnwidth * .75 # wide margins
    fig, ax = plt.subplots(*args, figsize=(width, width/aspect), **kwargs)
    return fig, ax

def fig_compact(aspect=1.25, *args, **kwargs):
    '''
    A compact but single subplot - for simple figures that stand alone
    '''
    width = columnwidth * .65 # wide margins
    fig, ax = plt.subplots(*args, figsize=(width, width/aspect), **kwargs)
    return fig, ax


def placeholder(text=''):
    ''' placeholder for a plot you didn't make yet '''
    fig, ax = fig1()
    rect = mpl.patches.Rectangle((0, 0), 1, 1, ec='red', fill=False, ls=':', lw=7)
    ax.add_patch(rect)
    ax.text(.5, .5, text, ha='center', va='center', color='red')
    ax.axis('off')


def fig_generating_scripts():
    '''
    Find which .py files import this file (plot_config)
    and then proceed to call savefig
    only works if this file (plot_config.py) is imported (rather than just run as a script)
    '''
    import ast
    config_dir = os.path.split(__file__)[0]
    config_dir = os.path.abspath(config_dir)
    print(f'Looking for figure scripts in {config_dir}')
    scripts = [f for f in os.listdir(config_dir) if f.endswith('.py')]

    def imports_this(script):
        with open(pjoin(config_dir, script), 'r', encoding='utf-8') as f:
            try:
                root = ast.parse(f.read())
            except:
                # there is something that is not python in the file
                print(f'Failed to parse {script}')
                return False
        for node in ast.iter_child_nodes(root):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if __name__ in n.name.split('.'):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if __name__ in node.module.split('.'):
                    return True
        return False

    def savesfig(script):
        with open(script, 'r', encoding='utf-8') as f:
        # does savefig actually get called at some point?
        # TODO: I think it will not catch plot_config.savefig..
            try:
                root = ast.parse(f.read())
            except:
                # there is something that is not python in the file
                print(f'Failed to parse {script}')
                return False

        for node in ast.walk(root):
            if isinstance(node, ast.Name) and node.id == 'savefig':
                return True

        #for node in ast.iter_child_nodes(root):
        #   if isinstance(node, ast.Expr):
        #       if isinstance(node.value, ast.Call):
        #           if hasattr(node.value.func, 'id'):
        #               if node.value.func.id == 'savefig':
        #                   return True
        return False

    scripts = list(filter(imports_this, scripts))
    scripts = list(filter(savesfig, scripts))
    return scripts


def make_all_figs(leave_open=False):
    '''
    Run all the fig generating scripts in the directory
    I am lazy so leave_open=False will close all figs,
    even ones you have open before calling this function
    '''
    import subprocess
    import runpy
    # if it imports this module, try to run it
    scripts = fig_generating_scripts()
    for script in scripts:
        print(f'Running {script}')
        #subprocess.call(script, shell=True)
        try:
            # Only works one time
            #importlib.import_module(script[:-3])
            runpy.run_path(script)
            plt.close('all')
        except Exception as e:
            print(e)
            #print(f'{script} failed at some point...')


def script_outputs(subdir='images'):
    '''
    I want to see how many figures are generated by each script
    mostly so I can find orphans and keep things clean
    '''
    all_files = os.listdir(subdir)
    image_files = fnmatch.filter(all_files, '*.png')

    print('Not considering these non-png files:')
    for f in all_files:
        if f not in image_files:
            print(f)

    print()
    scripts = fig_generating_scripts()

    config_dir = os.path.split(__file__)[0]
    config_dir = os.path.abspath(config_dir)
    all_scripts = [f for f in os.listdir(config_dir) if f.endswith('.py')]

    print()
    print('These scripts exist but do not import plot_config and call savefig():')
    for f in (set(all_scripts) - set(scripts)): print(f)

    accounted_for = set()

    print()
    print('This is the number of existing output images for each script:')
    for s in scripts:
        n = 0
        for im in image_files:
            if im.startswith(s[:-3]):
                n += 1
                accounted_for.add(im)
        print(f'{s}: {n}')

    print()
    print(f'Accounted for {len(accounted_for)} figures')
    # Any images that don't have a corresponding script?

    print()
    print('These png files do not start with the name of an existing script:')
    for f in (set(image_files) - accounted_for): print(f)


def latex_code(glob='*.png', subdir='images', exclude_existing=True, write_file=True):
    '''
    Get code to place all figures in latex document.
    easier to delete the ones you don't want than to put them all in by hand
    label with same label as filename - should simplify things
    NO rescaling. these are meant to be the right size unscaled.

    reads metadata from the png and puts it in the caption.
    (only png for now)
    this way, sample information or something else you thought important while creating the image can go directly into latex
    '''
    image_files = os.listdir(subdir)
    image_files = fnmatch.filter(image_files, glob)

    os.makedirs(latex_dir, exist_ok=True)

    # if write_file then we will write the output to this file
    tex_output_fn = 'Figures.tex'

    if exclude_existing:
        # look if any of the image files are already somewhere in the latex document and don't need to be placed
        tex_files = os.listdir(latex_dir)
        tex_files = fnmatch.filter(tex_files, '*.tex')
        tex_files = [f for f in tex_files if f != tex_output_fn]
        all_lines = []

        def placed(imf, text):
            if imf in all_text:
                print(f'{imf} already placed in {tf}')
                return True
            else:
                return False


        for tf in tex_files:
            with open(pjoin(latex_dir, tf), 'r', encoding='utf-8') as f:
                all_lines.extend(f.readlines())
            # easier to just concatenate..
            all_text = ''.join(all_lines)
            image_files = [imf for imf in image_files if not placed(imf, all_text)]

    allcode = []
    for f in image_files:
        if f.endswith('.png'):
            # see if I left any metadata there
            metadata = PIL.Image.open(pjoin(subdir,f)).info
            if 'description' in metadata:
                description = metadata['description']
            else:
                description = ''

        name = os.path.splitext(f)[0]
        #name_ = name.replace('_', '\_')
        # latex is a piece of shit and will crash in very strange ways if it encounters an underscore outside of math mode
        name_ = name.replace('_', ' ')
        description_ = description.replace('_', ' ')
        code = f'''\\begin{{figure}}[h]\n\\centering\n\\includegraphics[]{{../figures/{subdir}/{f}}}\n\\caption{{\\label{{fig:{name}}}\nAutomatic caption for {name_}. {description_}\n}}\n\\end{{figure}}\n\n'''.replace('\t', '')
        allcode.append(code)

    allcode = ''.join(allcode)
    print(allcode)

    if write_file:
        code = '% autogenerated by plot_config.py\n\\chapter{Figures}\n\maxdeadcycles=1000\n\n' + allcode
        fn = pjoin(latex_dir, tex_output_fn)
        with open(fn, 'w') as f:
            print(f'Writing {fn}')
            f.write(code)


def make_clean_image_folder(source='images', target='images_clean'):
    ''' for when you need to make a clean source, no unused graphics '''
    image_files = os.listdir(source)
    image_files = [i for i in image_files if os.path.isfile(os.path.join(source, i))]

    tex_files = os.listdir(latex_dir)
    tex_files = fnmatch.filter(tex_files, '*.tex')
    tex_files = [tx for tx in tex_files if tx != 'Figures.tex']
    all_lines = []

    for tf in tex_files:
        with open(pjoin(latex_dir, tf), 'r', encoding='utf-8') as f:
            all_lines.extend(f.readlines())

    # easier to just concatenate..
    all_text = ''.join(all_lines)

    def placed(imf, text):
        if imf in all_text:
            # bug: will also catch commented code, doesn't guarantee fig is in output pdf
            print(f'{imf} placed')
            return True
        else:
            return False

    image_files = [imf for imf in image_files if placed(imf, all_text)]

    os.makedirs(target, exist_ok=True)
    import shutil
    for fn in image_files:
        shutil.copyfile(os.path.join(source, fn), os.path.join(target, fn))


def uncommented_in_matplotlibrc(matplotlibrcfp='matplotlibrc'):
    '''
    matplotlibrc has almost everything commented because those are defaults
    this prints whatever is not commented
    '''
    with open(matplotlibrcfp) as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip()
        if l and not l.startswith('#'):
            print(l)


# Shared plotting utils

def inset(ax, width='40%', height='40%', loc=2, bbox_to_anchor=(0.05,0,1,1), bbox_transform=None, borderpad=1.0, **kwargs):
    if bbox_transform is None:
        bbox_transform = ax.transAxes
    # some standard inset, use smaller fonts and waste less space
    ins = inset_axes(ax, width=width, height=height, loc=loc, bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform, borderpad=borderpad, **kwargs)
    ins.set_xlabel('x', size='small', labelpad=1)
    ins.set_ylabel(r'y', size='small', labelpad=1)
    ins.tick_params(axis='both', which='major', labelsize='small', pad=1)
    return ins


def arrow(x0, y0, x1, y1, color='black', text='', ax=None, **kwargs):
    # plt.arrow is dumb.
    # here's a basic arrow.
    if ax is None:
        ax = plt.gca()
    kw = dict(arrowprops=dict(arrowstyle='->', color=color), ha='center', va='center', color=color)
    kw = {**kw, **kwargs}
    ax.annotate(text, (x1, y1), (x0, y0), **kw)


def dimarrow(x0, y0, x1, y1, color='black', bgcolor='white', text='', ha='center', va='center', dx=0, dy=0, bar=False, ax=None, arrowprops={}, fontsize='x-small', **kwargs):
    # for dimensioning
    # kind of a piece of crap. should use two separate arrows instead
    if ax is None:
        ax = plt.gca()
    kw = dict(arrowprops=dict(arrowstyle='<->', color=color, shrinkA=.1, shrinkB=.1, **arrowprops), ha='center', va='center',
              color=color, zorder=70)
    kw = {**kw, **kwargs}
    ax.annotate('', (x1, y1), (x0, y0), **kw)
    if bar:
        kw = deepcopy(kw) # avoids the most retarded bug ever jesus christ
        kw['arrowprops']['arrowstyle'] = '|-|' # too tall
        kw['arrowprops']['mutation_scale'] = 5
        kw['zorder'] = 65
        ax.annotate('', (x1, y1), (x0, y0), **kw)
    if text:
        bbox=dict(boxstyle='square,pad=0', fc=bgcolor, ec='none')
        ax.text((x0+x1)/2 + dx, (y0+y1)/2 + dy, text, ha=ha, va=va, bbox=bbox, zorder=71, size=fontsize, color=color)


def label_subplot(ax, label='A', x=-.01, y=1.1, color='black', **kwargs):
    '''
    Put a label outside of the axis, for referring to in the caption
    use this function to keep it consistent across figures
    '''
    txtargs = dict(transform=ax.transAxes, fontsize=14, color=color, fontweight='bold', va='top', ha='right')
    txtargs = {**txtargs, **kwargs}
    ax.text(x, y, label, **txtargs)


def plot_axes(ax=None):
    if ax is None:
        ax = plt.gca()
    #ax.grid()
    # This one does not control what the intersection looks like
    #ax.axvlines(0, linestyle='--', color='black', alpha=.5)
    #ax.axhlines(0, linestyle='--', color='black', alpha=.5)
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()

    lineprops = dict(color='black', alpha=.4, linestyle='--')
    ax.vlines(0, 0, yrange[0]*2, **lineprops)
    ax.vlines(0, 0, yrange[1]*2, **lineprops)
    ax.hlines(0, 0, xrange[0]*2, **lineprops)
    ax.hlines(0, 0, xrange[1]*2, **lineprops)

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)


def engformatter(axis='y', ax=None):
    if ax is None:
        ax = plt.gca()
    if axis.lower() == 'x':
        axis = ax.xaxis
    else:
        axis = ax.yaxis
    axis.set_major_formatter(mpl.ticker.EngFormatter())


def scale_axis_labels(scale=6, axis='y', ax=None):
    '''
    Scale the axis labels by factors of 10 without having to scale the data itself
    attempts to insert the metric prefix into the axis label
    don't attempt to apply it twice, it will mess up your axis label
    '''
    if ax is None:
        ax = plt.gca()

    prefix = mpl.ticker.EngFormatter()(10**-scale)[-1]
    #if prefix == 'µ':
    #    prefix = '$\mu$' # unicode looks different than tex

    axis = getattr(ax, axis+'axis')
    fmter = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*10**scale))
    axis.set_major_formatter(fmter)
    label =  axis.get_label_text()
    bracket = label.find('[')

    if (bracket > -1) and label[bracket+1] != prefix:
        axis.set_label_text(prefix.join((label[:bracket+1], label[bracket+1:])))

def μformat(axis='y', ax=None): scale_axis_labels(6, axis, ax)
def mformat(axis='y', ax=None): scale_axis_labels(3, axis, ax)
def nanoformat(axis='y', ax=None): scale_axis_labels(9, axis, ax)
def kformat(axis='y', ax=None): scale_axis_labels(-3, axis, ax)


# widgets/useful things for matplotlib lifestyle

def clipboard(text):
    # prints and puts code on the clipboard
    df = pd.DataFrame(text.split('\n'))
    df.to_clipboard(index=False,header=False)


def xylim(ax=None, clip=True):
    # return the command to set a plot xlim,ylim to the xlim and ylim of the current plot
    # also put it on the clipboard
    # got sick of repeating this over and over
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cmd = 'plt.gca().set_xlim({:.5e}, {:.5e})\nplt.gca().set_ylim({:.5e}, {:.5e})'.format(*xlim, *ylim)
    print(cmd)
    if clip:
        clipboard(cmd)


def shared_xylabel(fig, xlabel='what', ylabel='who', size='medium', **kwargs):
    # shared x-label that is actually centered relative to the axes
    # and works with constrained layout
    # hacky as hell because that was the only possible way at the time
    # needs matplotlib 3.4.0+
    axs = fig.get_axes()
    fig.supxlabel('dummy', size=size)
    fig.supylabel('dummy', size=size)
    plt.draw()
    plt.pause(.05)
    x0 = np.min([ax.get_position().x0 for ax in axs])
    x1 = np.max([ax.get_position().x1 for ax in axs])
    y0 = np.min([ax.get_position().y0 for ax in axs])
    y1 = np.max([ax.get_position().y1 for ax in axs])
    # This works because matplotlib is just a POS
    suplab1 = fig.supxlabel(xlabel, x=(x0 + x1)/2, size=size, **kwargs)
    suplab1._autopos = True
    y1 = np.max([ax.get_position().y1 for ax in axs])
    suplab2 = fig.supylabel(ylabel, y=(y0 + y1)/2, size=size, **kwargs)
    suplab2._autopos = True


def indicate_zoom(ax1, ax2, x0, x1, y0, y1, color='red', transform=None, lw=1):
    '''
    Show lines to indicate that ax2 is a zoom in of a region in ax1
    Default is data coordinates

    connecting lines might break if axes are inverted or something..?
    '''
    from matplotlib.patches import ConnectionPatch
    fig = ax1.get_figure()

    if transform is None:
        transform = ax1.transData

    x0,x1 = sorted((x0, x1))
    y0,y1 = sorted((y0, y1))

    # Put rectangle on ax1
    ax1.add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=0, transform=transform, edgecolor=color, lw=lw))

    # Change the color of the frame of ax2
    for k,spine in ax2.spines.items():
        spine.set_color(color)
        spine.set_lw(lw*1.5)

    # Get figure coordinates of the subregion
    def fig_coords(x, y):
        return fig.transFigure.inverted().transform(transform.transform((x,y)))
    fx0, fy0 = fig_coords(x0, y0)
    fx1, fy1 = fig_coords(x1, y1)
    from types import SimpleNamespace
    # should be the smaller square..
    small = SimpleNamespace(x0=fx0, y0=fy0, x1=fx1, y1=fy1)
    big = ax2.get_position()

    # this is not elegant.
    bigcorners = (None,(0,1),(1,1),(0,0),(1,0))
    smallcorners = (None,(x0,y1),(x1,y1),(x0,y0),(x1,y0))
    if (big.y0 <= small.y0) and (small.y1 <= big.y1):
        if big.x1 < small.x0:
            connect = (2,4,1,3) # big1, big2, small1, small2
        elif small.x1 < big.x0:
            connect = (1,3,2,4)
    elif (big.x0 <= small.x0) and (small.x1 <= big.x1):
        if big.y1 < small.y0:
            connect = (1,2,3,4)
        elif small.y1 < big.y0:
            connect = (3,4,1,2)
    else:
        con1 = ConnectionPatch((x0,y0), (0,0), coordsA=transform, coordsB='axes fraction', axesA=ax1, axesB=ax2, color=color, lw=lw)
        con2 = ConnectionPatch((x0,y1), (0,1), coordsA=transform, coordsB='axes fraction', axesA=ax1, axesB=ax2, color=color, lw=lw)
        con3 = ConnectionPatch((x1,y0), (1,0), coordsA=transform, coordsB='axes fraction', axesA=ax1, axesB=ax2, color=color, lw=lw)
        con4 = ConnectionPatch((x1,y1), (1,1), coordsA=transform, coordsB='axes fraction', axesA=ax1, axesB=ax2, color=color, lw=lw)
        fig.add_artist(con1)
        fig.add_artist(con2)
        fig.add_artist(con3)
        fig.add_artist(con4)

        a = small.x0 < big.x0
        b = small.x1 < big.x1
        c = small.y0 < big.y0
        d = small.y1 < big.y1

        con1.set_visible(a ^ c)
        con2.set_visible(a == d)
        con3.set_visible(b == c)
        con4.set_visible(b ^ d)
        return

    con1 = ConnectionPatch(smallcorners[connect[2]], bigcorners[connect[0]],
                            coordsA=transform, coordsB='axes fraction', axesA=ax1, axesB=ax2, color=color, lw=lw)
    con2 = ConnectionPatch(smallcorners[connect[3]], bigcorners[connect[1]],
                            coordsA=transform, coordsB='axes fraction', axesA=ax1, axesB=ax2, color=color, lw=lw)
    fig.add_artist(con1)
    fig.add_artist(con2)


class Cursor(AxesWidget):
    """
    Simple widget that prints out the points you click on while showing some lines
    Started with matplotlib.widgets.Cursor
    TODO: draw a line between points as you move cursor, sticking to points that you click
          put the list of points on the clipboard
    """
    def __init__(self, ax=None, horizOn=True, vertOn=True, useblit=True,
                 **lineprops):
        if ax is None:
            ax = plt.gca()
        AxesWidget.__init__(self, ax)
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)
        self.connect_event('button_press_event', self.onclick)
        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit
        self.n = 0
        lineprops = {'alpha':.5, 'linewidth':.5, 'color':'black', **lineprops}
        if self.useblit:
            lineprops['animated'] = True
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=False, **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], visible=False, **lineprops)
        self.background = None
        self.needclear = False

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)

    def onclick(self, event):
        print(f'x{self.n}, y{self.n} = {event.xdata:.3e}, {event.ydata:.3e}')
        self.ax.scatter(event.xdata, event.ydata, c='black', marker='x')
        self.n += 1

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return
        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)
        self._update()

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False
