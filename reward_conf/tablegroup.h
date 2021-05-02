#ifndef TABLEGROUP_H
#define TABLEGROUP_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QTableView;
class QHBoxLayout;
class QVBoxLayout;
QT_END_NAMESPACE

class TableGroup : public QWidget {
  Q_OBJECT

public:
  TableGroup(QWidget *parent = nullptr);

signals:
  void valueChanged(int value);

private:
  QLabel *queryLabel;
  QLineEdit *queryEdit;
  QTableView *resultView;
  QVBoxLayout *mainLayout;
  QHBoxLayout *queryLayout;
};

#endif
