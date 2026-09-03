import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox, QLineEdit, QColorDialog, QCheckBox, QApplication
from PyQt5.QtCore import Qt
import numpy as np
import sys
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
import copy


class PlotWidget(QWidget):
    """
    This class inherited from pyqtgraphs Plotwidget and MatplotlibWidget and adds additional compononets like:
        1) Cross-hair to view X, Y coordinates
        2) Changing plot-styles interactively
    """
    def __init__(self,parent=None,matplotlib=False):
        QWidget.__init__(self,parent)
        self.matplotlib=matplotlib
        self.mplPlotData={}
        self.mplErrorData={}
        self.xLabelFontSize=10
        self.yLabelFontSize=10
        self.titleFontSize=12
        self.xLabel='x'
        self.yLabel='y'
        self.title='Plot'            
        self.createPlotWidget()
        self.data={}
        self.dataErrPos={}
        self.dataErrNeg={}
        self.dataErr={}
        self.err={}
        self.data_num=0
        self.yerr={}
        self.fit={}
        self.linestyle={}   # per-curve Qt.PenStyle override (e.g. DashLine); None=solid
        # Optional callable(dname)->str to compute a friendly LEGEND label from a curve
        # key (the host sets this so the legend shows dataset names, not raw keys).
        self.legend_formatter=None
        
       
    def createPlotWidget(self):
        """
        Creates the plotWidget
        """
        self.vbLayout=QVBoxLayout(self)
        self.plotLayout=pg.LayoutWidget()
        self.vbLayout.addWidget(self.plotLayout)
        
        row=0
        col=0
        lineWidthLabel=QLabel('Line width')
        self.lineWidthLineEdit=QLineEdit('2')
        self.lineWidthLineEdit.returnPressed.connect(self.updatePlot)
        pointSizeLabel=QLabel('Point size')
        self.pointSizeLineEdit=QLineEdit('5')
        self.pointSizeLineEdit.returnPressed.connect(self.updatePlot)
        self.bgCheckBox=QCheckBox('White BG')
        self.bgCheckBox.stateChanged.connect(self.bgCheckBoxChanged)
        self.errorbarCheckBox=QCheckBox('Errorbar')
        self.errorbarCheckBox.stateChanged.connect(self.errorbarChanged)
        # Data line toggle: when OFF (default) DATA curves are drawn as markers only
        # (no connecting line), so they don't read as a second fit curve when a fit is
        # overlaid. When ON, data also gets its connecting line. Fit curves are always
        # lines regardless of this toggle.
        self.dataLineCheckBox=QCheckBox('Data line')
        self.dataLineCheckBox.setToolTip('Draw a connecting line through DATA points '
                                         '(fits are always lines). Off = markers only.')
        self.dataLineCheckBox.stateChanged.connect(self.dataLineChanged)
        self.plotLayout.addWidget(lineWidthLabel,row=row,col=col)
        col+=1
        self.plotLayout.addWidget(self.lineWidthLineEdit,row=row,col=col)
        col+=1
        self.plotLayout.addWidget(pointSizeLabel,row=row,col=col)
        col+=1
        self.plotLayout.addWidget(self.pointSizeLineEdit,row=row,col=col)
        col+=1
        self.plotLayout.addWidget(self.bgCheckBox,row=row,col=col)
        col+=1
        self.plotLayout.addWidget(self.errorbarCheckBox,row=row,col=col)
        col+=1
        self.plotLayout.addWidget(self.dataLineCheckBox,row=row,col=col)
        col=0
        row+=1
        if self.matplotlib:
            self.plotWidget=MatplotlibWidget()
            self.subplot=self.plotWidget.getFigure().add_subplot(111)
            self.plotWidget.fig.set_tight_layout(True)
            self.plotWidget.draw()
        else:
            self.plotWidget=pg.PlotWidget()
            self.plotWidget.getPlotItem().vb.scene().sigMouseMoved.connect(self.mouseMoved)
            self.legendItem=pg.LegendItem(offset=(0.0,1.0))
            self.legendItem.setParentItem(self.plotWidget.getPlotItem())
            
        self.plotLayout.addWidget(self.plotWidget,row=row,col=col,colspan=6)
        row+=1
        col=0 
        self.crosshairLabel=QLabel(u'X={: .5f} , y={: .5f}'.format(0.0,0.0))                                 
        self.xLogCheckBox=QCheckBox('LogX')
        self.xLogCheckBox.setTristate(False)
        self.xLogCheckBox.stateChanged.connect(self.updatePlot)
        self.yLogCheckBox=QCheckBox('LogY')
        self.yLogCheckBox.setTristate(False)
        self.yLogCheckBox.stateChanged.connect(self.updatePlot)
        if not self.matplotlib:
            self.plotLayout.addWidget(self.crosshairLabel,row=row,col=col,colspan=4)
        self.plotLayout.addWidget(self.xLogCheckBox,row=row,col=4)
        self.plotLayout.addWidget(self.yLogCheckBox,row=row,col=5)
        
        
        
    def bgCheckBoxChanged(self):
        if self.bgCheckBox.isChecked():
            self.plotWidget.setBackground('w')
            self.plotWidget.getAxis('left').setPen('k')
            self.plotWidget.getAxis('left').setTextPen('k')
            self.plotWidget.getAxis('bottom').setPen('k')
            self.plotWidget.getAxis('bottom').setTextPen('k')
        else:
            self.plotWidget.setBackground('k')
            self.plotWidget.getAxis('left').setPen('w')
            self.plotWidget.getAxis('left').setTextPen('w')
            self.plotWidget.getAxis('bottom').setPen('w')
            self.plotWidget.getAxis('bottom').setTextPen('w')


    def mouseMoved(self,pos):
        try:
            pointer=self.plotWidget.getPlotItem().vb.mapSceneToView(pos)
            x,y=pointer.x(),pointer.y()
            if self.plotWidget.getPlotItem().ctrl.logXCheck.isChecked():
                x=10**x
            if self.plotWidget.getPlotItem().ctrl.logYCheck.isChecked():
                y=10**y
            self.crosshairLabel.setText('X={: 10.5f}, Y={: 10.5e}'.format(x,y))
            # if x>1e-3 and y>1e-3:
            #     self.crosshairLabel.setText(u'X={: .5f} , Y={: .5f}'.format(x,y))
            # if x<1e-3 and y>1e-3:
            #     self.crosshairLabel.setText(u'X={: .3e} , Y={: .5f}'.format(x,y))
            # if x>1e-3 and y<1e-3:
            #     self.crosshairLabel.setText(u'X={: .5f} , Y={: .3e}'.format(x,y))
            # if x<1e-3 and y<1e-3:
            #     self.crosshairLabel.setText(u'X={: .3e} , Y={: .3e}'.format(x,y))
        except:
            pass
                
        #self.crosshairLabel.setText(u'X=%+0.5f, Y=%+0.5e'%(x,y))
        
        
    def _legend_text(self,dname):
        """Friendly legend label for a curve key via the host-supplied formatter;
        falls back to the raw key."""
        try:
            if callable(self.legend_formatter):
                lbl=self.legend_formatter(dname)
                if lbl:
                    return lbl
        except Exception:
            pass
        return dname

    def _pen_for(self,dname,color):
        """Build the line pen for a curve, honouring the fit flag, the 'Data line'
        toggle and any per-curve linestyle. Returns None for DATA curves when the line
        is suppressed (markers-only) -- pyqtgraph then draws symbols with no line."""
        try:
            width=float(self.lineWidthLineEdit.text())
        except Exception:
            width=2.0
        style=self.linestyle.get(dname) or Qt.SolidLine
        is_fit=self.fit.get(dname,False)
        if not is_fit:
            # DATA: line only when the toggle is on and width != 0
            if (not self.dataLineCheckBox.isChecked()) or width==0:
                return None
        return pg.mkPen(color=color,width=width,style=style)

    def dataLineChanged(self):
        """Re-pen every curve honouring the DATA connecting-line toggle, then re-plot.
        (Plot() alone re-adds existing items without recomputing pens, so the pen must
        be rebuilt here for the toggle to take effect.)"""
        for dname in list(self.data.keys()):
            try:
                color=self.data[dname].opts['symbolPen'].color()
            except Exception:
                try:
                    color=self.data[dname].opts['pen'].color()
                except Exception:
                    continue
            pen=self._pen_for(dname,color)
            if pen is None:
                self.data[dname].opts['pen']=None
                self.data[dname].updateItems()
            else:
                self.data[dname].setPen(pen)
        try:
            self.Plot(self.selDataNames)
        except Exception:
            pass

    def add_data(self,x,y,yerr=None,name=None,fit=False,color=None,linestyle=None):
        """
        Adds data into the plot where:
        x=Array of x-values
        y=Array of y-values
        yerr=Array of yerr-values. If None yerr will be set to sqrt(y)
        name=any string to be used for the key to put the data
        fit= True if the data corresponds to a fit
        linestyle= optional Qt.PenStyle (e.g. Qt.DashLine) for the line; None=solid.
                   Used to distinguish overlaid curves (e.g. individual vs joint fit).
        """
        if not (isinstance(yerr,list) or isinstance(yerr,np.ndarray)):
            yerr=np.ones_like(y)
        if len(x)==len(y) and len(y)==len(yerr):
            if name is None:
                dname=str(self.data_num)
            else:
                dname=name
            if np.all(yerr==1):
                self.yerr[dname]=False
            else:
                self.yerr[dname]=True
            self.fit[dname]=fit
            self.linestyle[dname]=linestyle
            if dname in self.data.keys():
                if color is None:
                    color=self.data[dname].opts['symbolPen'].color()
                pen=self._pen_for(dname,color)
                symbol='o'
                if self.fit[dname]:
                    symbol=None
                self.data[dname].setData(x,y,pen=pen,symbol=symbol,symbolSize=float(self.pointSizeLineEdit.text()),symbolPen=pg.mkPen(color=color),symbolBrush=pg.mkBrush(color=color))
                #self.data[dname].setPen(pg.mkPen(color=pg.intColor(np.random.choice(range(0,210),1)[0]),width=int(self.lineWidthLineEdit.text())))
                #if self.errorbarCheckBox.isChecked():
                # self.dataErrPos[dname].setData(x,np.where(y+yerr/2.0>0,y+yerr/2.0,y))
                # self.dataErrNeg[dname].setData(x,np.where(y-yerr/2.0>0,y-yerr/2.0,y))
                self.err[dname]= yerr
                self.dataErr[dname].setData(x=x, y=y, top=self.err[dname], bottom=self.err[dname], pen='w')# beam=min(np.abs(x))*0.01*float(self.pointSizeLineEdit.text()),pen='w')
            #self.dataErr[dname].setCurves(self.dataErrPos[dname],self.dataErrNeg[dname])
            else:
                if color is None:
                    # deterministic, well-separated palette (Tableau-10) instead of a
                    # random color, so color-less curves are distinct and stable.
                    _pal=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
                    _ct=getattr(self,'_palette_ct',0)
                    color=_pal[_ct % len(_pal)]
                    self._palette_ct=_ct+1
                #color=self.data[dname].opts['pen'].color()
                pen=self._pen_for(dname,color)
                symbol='o'
                if self.fit[dname]:
                    symbol=None
                self.data[dname]=pg.PlotDataItem(x,y,pen=pen,symbol=symbol,symbolSize=float(self.pointSizeLineEdit.text()),symbolPen=pg.mkPen(color=color),symbolBrush=pg.mkBrush(color=color))
                self.dataErr[dname] = pg.ErrorBarItem()
                self.err[dname]=yerr
                self.dataErr[dname].setData(x=x,y=y,top=self.err[dname],bottom=self.err[dname], pen='w')# beam=min(np.abs(x))*0.01*float(self.pointSizeLineEdit.text()),pen='w')
                self.data[dname].curve.setClickable(True,width=10)
                self.data[dname].sigClicked.connect(self.colorChanged)
                #if self.errorbarCheckBox.isChecked():
                # self.dataErrPos[dname]=pg.PlotDataItem(x,np.where(y+yerr/2.0>0,y+yerr/2.0,y))
                # self.dataErrNeg[dname]=pg.PlotDataItem(x,np.where(y-yerr/2.0>0,y-yerr/2.0,y))
                #self.dataErr[dname]=pg.FillBetweenItem(curve1=self.dataErrPos[dname],curve2=self.dataErrNeg[dname],brush=pg.mkBrush(color=pg.hsvColor(1.0,sat=0.0,alpha=0.2)))
                self.data_num+=1
                #if len(x)>1:
                self.Plot([dname])
            return True
        else:
            QMessageBox.warning(self,'Data error','The dimensions of x, y or yerr are not matching',QMessageBox.Ok)
            return False
            
    def colorChanged(self,item):
        """
        Color of the item changed
        """
        color=item.opts['symbolPen'].color()
        newcolor=QColorDialog.getColor(initial=color)
        if newcolor.isValid():
            #if self.lineWidthLineEdit.text()!='0':
            item.setPen(pg.mkPen(color=newcolor,width=int(self.lineWidthLineEdit.text())))
            if self.pointSizeLineEdit.text()!='0':
                item.setSymbolBrush(pg.mkBrush(color=newcolor))
                item.setSymbolPen(pg.mkPen(color=newcolor))
    
    def errorbarChanged(self):
        """
        Updates the plot checking the Errorbar is checked or not
        """
        try:
            self.Plot(self.selDataNames)
        except:
            pass
            
        
    def Plot(self,datanames):
        """
        Plots all the data in the memory with errorbars where:
            datanames is the list of datanames
        """
        self.selDataNames=datanames
        if self.matplotlib: #Plotting with matplotlib
            names=list(self.mplPlotData.keys())
            for name in names:
                if name not in self.selDataNames:
                    self.mplPlotData[name].remove()
                    del self.mplPlotData[name]
                    try:
                        self.mplErrorData[name].remove()
                        del self.mplErrorData[name]
                    except:
                        pass
            self.xLabel=self.subplot.get_xlabel()
            self.yLabel=self.subplot.get_ylabel()
            self.title=self.subplot.get_title()
            #self.subplot.axes.cla()
            for dname in self.selDataNames:
                if self.fit[dname]:
                    plot_type = '-'
                elif self.dataLineCheckBox.isChecked():
                    plot_type = '.-'
                else:
                    plot_type = '.'
                if self.errorbarCheckBox.checkState()==Qt.Checked:
                    try:
                        self.mplPlotData[dname].set_xdata(self.data[dname].xData)
                        self.mplPlotData[dname].set_ydata(self.data[dname].yData)
                        self.mplPlotData[dname].set_markersize(int(self.pointSizeLineEdit.text()))
                        self.mplPlotData[dname].set_linewidth(int(self.lineWidthLineEdit.text()))

                        self.mplErrorData[dname].set_segments(np.array([[x,yt],[x,yb]]) for x,yt,yb in zip(self.data[dname].xData,self.dataErrPos[dname].yData,self.dataErrNeg[dname].yData))
                        self.mplErrorData[dname].set_linewidth(2)
                    except:
                        ln,err,bar =self.subplot.errorbar(self.data[dname].xData,self.data[dname].yData,xerr=None,yerr=self.dataErr[dname].opts['top']*2,fmt=plot_type,markersize=int(self.pointSizeLineEdit.text()),linewidth=int(self.lineWidthLineEdit.text()),label=self._legend_text(dname))
                        self.mplPlotData[dname]=ln
                        self.mplErrorData[dname],=bar
                else:
                    try:
                        self.mplPlotData[dname].set_xdata(self.data[dname].xData)
                        self.mplPlotData[dname].set_ydata(self.data[dname].yData)
                        self.mplPlotData[dname].set_markersize(int(self.pointSizeLineEdit.text()))
                        self.mplPlotData[dname].set_linewidth(int(self.lineWidthLineEdit.text()))
                    except:
                        self.mplPlotData[dname], =self.subplot.plot(self.data[dname].xData,self.data[dname].yData,plot_type,markersize=int(self.pointSizeLineEdit.text()),linewidth=int(self.lineWidthLineEdit.text()),label=self._legend_text(dname))
            if self.xLogCheckBox.checkState()==Qt.Checked:
                self.subplot.set_xscale('log')
            else:
                self.subplot.set_xscale('linear')
            if self.yLogCheckBox.checkState()==Qt.Checked:
                self.subplot.set_yscale('log')
            else:
                self.subplot.set_yscale('linear')
            self.subplot.set_xlabel(self.xLabel,fontsize=self.xLabelFontSize)
            self.subplot.set_ylabel(self.yLabel,fontsize=self.yLabelFontSize)
            self.subplot.set_title(self.title,fontsize=self.titleFontSize)
#            try:
#                self.leg.draggable()
#            except:
            self.leg=self.subplot.legend()
            self.leg.set_draggable(True)
            self.plotWidget.fig.set_tight_layout(True)
            self.plotWidget.draw()
                
        else:
            self.plotWidget.plotItem.setLogMode(x=False,y=False)
            self.plotWidget.clear()
            # Rebuild the legend from scratch each redraw. removeItem(name) matches by the
            # DISPLAYED label, which now differs from the curve key (legend_formatter), so
            # per-name removal silently failed and legend entries piled up (every curve,
            # duplicated, out of order). clear() is authoritative; fall back to removing by
            # the item reference for older pyqtgraph without LegendItem.clear().
            try:
                self.legendItem.clear()
            except Exception:
                for _n in list(self.data.keys()):
                    try:
                        self.legendItem.removeItem(self.data[_n])
                    except Exception:
                        pass
            xlog_res=True
            ylog_res=True
            for dname in self.selDataNames:
                if np.all(self.data[dname].yData==0) and self.yLogCheckBox.checkState()==Qt.Checked: #This step is necessary for checking the zero values
                    QMessageBox.warning(self,'Zero error','All the yData are zeros. So Cannot plot Logarithm of yData for %s'%dname,QMessageBox.Ok)
                    ylog_res=ylog_res and False
                    if not ylog_res:
                        self.yLogCheckBox.stateChanged.disconnect(self.updatePlot)
                        self.yLogCheckBox.setCheckState(Qt.Unchecked)
                        self.yLogCheckBox.stateChanged.connect(self.updatePlot)
                if np.all(self.data[dname].xData==0) and self.xLogCheckBox.checkState()==Qt.Checked:
                    QMessageBox.warning(self,'Zero error','All the xData are zeros. So Cannot plot Logarithm of xData for %s'%dname,QMessageBox.Ok)
                    xlog_res=xlog_res and False
                    if not xlog_res:
                        self.xLogCheckBox.stateChanged.disconnect(self.updatePlot)
                        self.xLogCheckBox.setCheckState(Qt.Unchecked)
                        self.xLogCheckBox.stateChanged.connect(self.updatePlot)
                self.plotWidget.addItem(self.data[dname])
                if self.errorbarCheckBox.isChecked() and self.yerr[dname]:
                    x=self.data[dname].xData
                    y=self.data[dname].yData
                    top=copy.copy(self.err[dname])
                    bottom= copy.copy(self.err[dname])
                    if self.xLogCheckBox.checkState() == Qt.Checked:
                         x=np.log10(x)
                    if self.yLogCheckBox.checkState() == Qt.Checked:
                        top=np.log10(1+top/y)
                        bottom=np.log10(y/(y-bottom))
                        y = np.log10(y)
                    self.dataErr[dname].setData(x=x,y=y,top=top,bottom=bottom,pen='w')#beam=min(np.abs(x))*0.01*float(self.pointSizeLineEdit.text()),pen='w')
                    self.plotWidget.addItem(self.dataErr[dname])
                    # self.plotWidget.addItem(self.dataErrPos[dname])
                    # self.plotWidget.addItem(self.dataErrNeg[dname])
                    #self.plotWidget.addItem(self.dataErr[dname])
                    #self.dataErr[dname].setCurves(self.dataErrPos[dname],self.dataErrNeg[dname])
                self.legendItem.addItem(self.data[dname],self._legend_text(dname))
            if self.xLogCheckBox.checkState()==Qt.Checked:
                self.plotWidget.plotItem.setLogMode(x=True)
            else:
                self.plotWidget.plotItem.setLogMode(x=False)
            if self.yLogCheckBox.checkState()==Qt.Checked:
                self.plotWidget.plotItem.setLogMode(y=True)
            else:
                self.plotWidget.plotItem.setLogMode(y=False)

                
    def remove_data(self,datanames):
        for dname in datanames:
            if self.matplotlib:
                self.mplPlotData[dname].remove()
                del self.mplPlotData[dname]
                if self.errorbarCheckBox.isChecked() and self.yerr[dname]:
                    self.mplErrorData[dname].remove()
                    del self.mplErrorData[dname]
            else:
                self.plotWidget.removeItem(self.data[dname])
                self.legendItem.removeItem(dname)
                if self.errorbarCheckBox.isChecked() and self.yerr[dname]:
                    self.plotWidget.removeItem(self.dataErr[dname])
                    # self.plotWidget.removeItem(self.dataErrPos[dname])
                    # self.plotWidget.removeItem(self.dataErrNeg[dname])
            del self.data[dname]
            del self.dataErr[dname]
            # del self.dataErrPos[dname]
            # del self.dataErrNeg[dname]
            
            
    def updatePlot(self):
        try:
            for dname in self.selDataNames:
                # curve color = the symbol's color (data + fit share a color)
                try:
                    color=self.data[dname].opts['symbolPen'].color()
                except Exception:
                    color=self.data[dname].opts['pen'].color()
                pen=self._pen_for(dname,color)   # honours 'Data line' toggle + linestyle
                if pen is None:
                    self.data[dname].opts['pen']=None
                    self.data[dname].updateItems()
                else:
                    self.data[dname].setPen(pen)
                self.data[dname].opts['symbol']='o'
                if self.fit[dname]:
                    self.data[dname].setSymbolSize(0)
                else:
                    self.data[dname].setSymbolSize(float(self.pointSizeLineEdit.text()))
            self.Plot(self.selDataNames)
        except:
            QMessageBox.warning(self,'Data Error','No data to plot',QMessageBox.Ok)
            
    def setXLabel(self,label,fontsize=4):
        """
        sets the X-label of the plot
        """
        self.xLabel=label
        self.xLabelFontSize=fontsize
        if self.matplotlib:
            self.subplot.set_xlabel(label,fontsize=fontsize)
            self.plotWidget.draw()
        else:
            self.plotWidget.getPlotItem().setLabel('bottom','<font size='+str(fontsize)+'>'+label+'</font>')
            
    def setYLabel(self,label,fontsize=4):
        """
        sets the y-label of the plot
        """
        self.yLabel=label
        self.yLabelFontSize=fontsize
        if self.matplotlib:
            self.subplot.set_ylabel(label,fontsize=fontsize)
            self.plotWidget.draw()
        else:
            self.plotWidget.getPlotItem().setLabel('left','<font size='+str(fontsize)+'>'+label+'</font>')        
            
            
    def setTitle(self,title,fontsize=6):
        """
        Sets the y-label of the plot
        """
        self.title=title
        self.titleFontSize=fontsize
        if self.matplotlib:
            self.subplot.set_title(title,fontsize=fontsize)
            self.plotWidget.draw()
        else:
            self.plotWidget.getPlotItem().setTitle(title='<font size='+str(fontsize)+'>'+title+'</font>')

    def addROI(self,values=(0,1),orientation='horizontal',movable=True, minmax_widget=None, min_widget=None, max_widget=None):
        if orientation=='vertical':
            self.roi=pg.LinearRegionItem(values=values,orientation=pg.LinearRegionItem.Vertical,movable=movable)
        else:
            self.roi = pg.LinearRegionItem(values=values, orientation=pg.LinearRegionItem.Horizontal,movable=movable)
        self.plotWidget.addItem(self.roi)
        return self.roi

 
    
if __name__=='__main__':
    app=QApplication(sys.argv)
    w=PlotWidget(matplotlib=False)
    x=np.arange(0,np.pi,0.01)
    y=np.sin(x)
    yerr=0.1*y
    w.add_data(x,y,yerr=yerr,name='sin')
    w.setXLabel('x',fontsize=15)
    w.setYLabel('y',fontsize=15)
    w.setTitle('My Plot',fontsize=15)
    w.setWindowTitle('Plot Widget')
    w.setGeometry(100,100,1000,800)
    
    w.show()
    sys.exit(app.exec_())