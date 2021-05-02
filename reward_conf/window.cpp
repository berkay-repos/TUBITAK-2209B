#include "window.h"
#include "slidersgroup.h"
#include "tablegroup.h"

#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QStackedWidget>

Window::Window(QWidget *parent)
    : QWidget(parent)
{
  horizontalSliders = new SlidersGroup;
  verticalSliders = new TableGroup;

  QVBoxLayout *layout = new QVBoxLayout;
  layout->addWidget(verticalSliders);
  layout->addWidget(horizontalSliders);
  setLayout(layout);

  setWindowTitle(tr("Reward Configuration"));
}
