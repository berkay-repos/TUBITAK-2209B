#include "slidersgroup.h"

#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDial>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollBar>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QtMath>

SlidersGroup::SlidersGroup(QWidget *parent) : QWidget(parent) {
  controlsGroup = new QGroupBox(tr("Parameter Configuration"));

  Reward1 = new QLabel(tr("Reward conf 1: "));
  Reward2 = new QLabel(tr("Reward conf 2: "));
  stackedWidget = new QStackedWidget;
  stackedWidget->addWidget(Reward1);
  stackedWidget->addWidget(Reward2);

  ATALabel = new QLabel(tr("ATA degree"));
  AALabel = new QLabel(tr("AA degree"));
  distanceLabel = new QLabel(tr("Distance"));
  kLabel = new QLabel(tr("k"));

  invertedAppearance = new QCheckBox(tr("Inverted appearance"));
  invertedKeyBindings = new QCheckBox(tr("Inverted key bindings"));

  ATASpinBox = new QSpinBox;
  ATASpinBox->setRange(-360, 360);
  ATASpinBox->setSingleStep(1);

  AASpinBox = new QSpinBox;
  AASpinBox->setRange(-360, 360);
  AASpinBox->setSingleStep(1);

  distanceSpinBox = new QSpinBox;
  distanceSpinBox->setRange(25, 800);
  distanceSpinBox->setSingleStep(155);

  kSpinBox = new QDoubleSpinBox;
  kSpinBox->setRange(.1, 100);
  kSpinBox->setSingleStep(.1);
  kSpinBox->setDecimals(1);

  orientationCombo = new QComboBox;
  orientationCombo->addItem(tr("Reward 1"));
  orientationCombo->addItem(tr("Reward 2"));

  QGridLayout *controlsLayout = new QGridLayout;
  controlsLayout->addWidget(ATALabel, 0, 0);
  controlsLayout->addWidget(AALabel, 1, 0);
  controlsLayout->addWidget(distanceLabel, 2, 0);
  controlsLayout->addWidget(kLabel, 3, 0);
  controlsLayout->addWidget(ATASpinBox, 0, 1);
  controlsLayout->addWidget(AASpinBox, 1, 1);
  controlsLayout->addWidget(distanceSpinBox, 2, 1);
  controlsLayout->addWidget(kSpinBox, 3, 1);
  controlsLayout->addWidget(invertedAppearance, 0, 2);
  controlsLayout->addWidget(invertedKeyBindings, 1, 2);
  controlsLayout->addWidget(orientationCombo, 4, 0, 1, 3);
  controlsGroup->setLayout(controlsLayout);

  slider = new QSlider(Qt::Horizontal);
  slider->setFocusPolicy(Qt::StrongFocus);
  slider->setTickPosition(QSlider::TicksBothSides);
  slider->setMinimum(25);
  slider->setMaximum(800);
  slider->setTickInterval(155);
  slider->setSingleStep(1);

  scrollBar = new QScrollBar(Qt::Horizontal);
  scrollBar->setFocusPolicy(Qt::StrongFocus);
  scrollBar->setMinimum(0.1);
  scrollBar->setMaximum(100);

  dial = new QDial;
  dial->setFocusPolicy(Qt::StrongFocus);
  dial->setMinimum(-360);
  dial->setMaximum(360);

  dial2 = new QDial;
  dial2->setFocusPolicy(Qt::StrongFocus);
  dial2->setMinimum(-360);
  dial2->setMaximum(360);

  slidergroup = new QGroupBox(tr("Controls"));

  QVBoxLayout *slidersLayout = new QVBoxLayout;
  slidersLayout->addWidget(dial);
  slidersLayout->addWidget(dial2);
  slidersLayout->addWidget(slider);
  slidersLayout->addWidget(scrollBar);
  slidersLayout->addWidget(stackedWidget);
  slidergroup->setLayout(slidersLayout);

  QHBoxLayout *layout = new QHBoxLayout;
  layout->addWidget(controlsGroup);
  layout->addWidget(slidergroup);
  setLayout(layout);

  connect(dial, &QDial::valueChanged, this, &SlidersGroup::setValue);
  connect(dial2, &QDial::valueChanged, this, &SlidersGroup::setValue);
  connect(slider, &QSlider::valueChanged, this, &SlidersGroup::setValue);
  connect(scrollBar, &QScrollBar::valueChanged, this, &SlidersGroup::setValue);
  connect(orientationCombo, qOverload<int>(&QComboBox::activated),
          stackedWidget, &QStackedWidget::setCurrentIndex);
  connect(kSpinBox, qOverload<double>(&QDoubleSpinBox::valueChanged), scrollBar,
          &QScrollBar::setValue);
  connect(scrollBar, qOverload<int>(&QScrollBar::valueChanged), kSpinBox,
          &QDoubleSpinBox::setValue);
  connect(slider, qOverload<int>(&QSlider::valueChanged), distanceSpinBox,
          &QSpinBox::setValue);
  connect(distanceSpinBox, qOverload<int>(&QSpinBox::valueChanged), slider,
          &QSlider::setValue);
  connect(dial2, qOverload<int>(&QDial::valueChanged), AASpinBox,
          &QSpinBox::setValue);
  connect(AASpinBox, qOverload<int>(&QSpinBox::valueChanged), dial2,
          &QDial::setValue);
  connect(dial, qOverload<int>(&QDial::valueChanged), ATASpinBox,
          &QSpinBox::setValue);
  connect(ATASpinBox, qOverload<int>(&QSpinBox::valueChanged), dial,
          &QDial::setValue);
  connect(invertedAppearance, &QCheckBox::toggled, this,
          &SlidersGroup::invertAppearance);
  connect(invertedAppearance, &QCheckBox::toggled, this,
          &SlidersGroup::invertAppearance);
  connect(invertedKeyBindings, &QCheckBox::toggled, this,
          &SlidersGroup::invertKeyBindings);
  connect(invertedKeyBindings, &QCheckBox::toggled, this,
          &SlidersGroup::invertKeyBindings);
}

void SlidersGroup::setValue() {
  double D, O, S;
  D = qExp(
      (-qFabs(static_cast<double>(distanceSpinBox->value()) - (300 + 25) / 2)) /
      (180 * kSpinBox->value()));
  O = 1 - (qFabs((static_cast<double>(ATASpinBox->value())) +
                 qFabs(static_cast<double>(AASpinBox->value()))) /
           180);
  S = (((1 - static_cast<double>(AASpinBox->value()) / 180) +
        (1 - static_cast<double>(ATASpinBox->value()) / 180)) /
       2) *
      qExp(-abs(static_cast<double>(distanceSpinBox->value()) - 120) /
           (120 * kSpinBox->value()));

  Reward1->setText("Reward conf 1: " + QString::number(D * O / 100));
  Reward2->setText("Reward conf 2: " + QString::number((1 - 0.8) * S));
}

void SlidersGroup::invertAppearance(bool invert)
{
  slider->setInvertedAppearance(invert);
  scrollBar->setInvertedAppearance(invert);
  dial->setInvertedAppearance(invert);
  dial2->setInvertedAppearance(invert);
}

void SlidersGroup::invertKeyBindings(bool invert)
{
  slider->setInvertedControls(invert);
  scrollBar->setInvertedControls(invert);
  dial->setInvertedControls(invert);
  dial2->setInvertedControls(invert);
}
