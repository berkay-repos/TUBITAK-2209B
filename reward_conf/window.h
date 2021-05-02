#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QSpinBox;
class QStackedWidget;
QT_END_NAMESPACE
class SlidersGroup;
class TableGroup;

class Window : public QWidget
{
    Q_OBJECT

public:
    Window(QWidget *parent = nullptr);

  private:
    SlidersGroup *horizontalSliders;
    TableGroup *verticalSliders;
    QStackedWidget *stackedWidget;

    QGroupBox *controlsGroup;
    QLabel *minimumLabel;
    QLabel *maximumLabel;
    QLabel *valueLabel;
    QCheckBox *invertedAppearance;
    QCheckBox *invertedKeyBindings;
    QSpinBox *minimumSpinBox;
    QSpinBox *maximumSpinBox;
    QSpinBox *valueSpinBox;
    QComboBox *orientationCombo;
};

#endif
