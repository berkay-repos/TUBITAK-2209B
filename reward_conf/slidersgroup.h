#ifndef SLIDERSGROUP_H
#define SLIDERSGROUP_H

#include <QGroupBox>
#include <QWidget>

QT_BEGIN_NAMESPACE
class QDial;
class QScrollBar;
class QSlider;
class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QSpinBox;
class QDoubleSpinBox;
class QStackedWidget;
QT_END_NAMESPACE
class SlidersGroup;

class SlidersGroup : public QWidget {
  Q_OBJECT

public:
  SlidersGroup(QWidget *parent = nullptr);

signals:
  void valueChanged(int value);

public slots:
  void setValue();
  void invertAppearance(bool invert);
  void invertKeyBindings(bool invert);

private:
  QSlider *slider;
  QScrollBar *scrollBar;
  QDial *dial;
  QDial *dial2;

  SlidersGroup *horizontalSliders;
  SlidersGroup *verticalSliders;
  QStackedWidget *stackedWidget;

  QGroupBox *controlsGroup;
  QGroupBox *slidergroup;
  QLabel *ATALabel;
  QLabel *AALabel;
  QLabel *distanceLabel;
  QLabel *kLabel;
  QLabel *Reward1;
  QLabel *Reward2;
  QCheckBox *invertedAppearance;
  QCheckBox *invertedKeyBindings;
  QSpinBox *ATASpinBox;
  QSpinBox *AASpinBox;
  QSpinBox *distanceSpinBox;
  QDoubleSpinBox *kSpinBox;
  QComboBox *orientationCombo;
};

#endif
