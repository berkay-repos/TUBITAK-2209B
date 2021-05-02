#include "tablegroup.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QTableView>
#include <QVBoxLayout>
#include <QtWidgets>

TableGroup::TableGroup(QWidget *parent) : QWidget(parent) {
  queryLabel = new QLabel(tr("Query:"));
  queryEdit = new QLineEdit();
  resultView = new QTableView();

  queryLayout = new QHBoxLayout();
  queryLayout->addWidget(queryLabel);
  queryLayout->addWidget(queryEdit);

  mainLayout = new QVBoxLayout();
  mainLayout->addLayout(queryLayout);
  mainLayout->addWidget(resultView);
  setLayout(mainLayout);

  QStandardItemModel model;
  model.setHorizontalHeaderLabels({tr("Config"), tr("Office")});

  const QStringList rows[] = {
      QStringList{QStringLiteral("Verne Nilsen "), QStringLiteral("123")},
      QStringList{QStringLiteral("Carlos Tang "), QStringLiteral("77 ")},
      QStringList{QStringLiteral("Bronwyn Hawcroft "), QStringLiteral("119")},
      QStringList{QStringLiteral("Alessandro Hanssen"), QStringLiteral("32 ")},
      QStringList{QStringLiteral("Andrew John Bakken"), QStringLiteral("54 ")},
      QStringList{QStringLiteral("Vanessa Weatherley"), QStringLiteral("85 ")},
      QStringList{QStringLiteral("Rebecca Dickens "), QStringLiteral("17 ")},
      QStringList{QStringLiteral("David Bradley "), QStringLiteral("42 ")},
      QStringList{QStringLiteral("Knut Walters "), QStringLiteral("25 ")},
      QStringList{QStringLiteral("Andrea Jones "), QStringLiteral("34 ")}};

  QList<QStandardItem *> items;
  for (const QStringList &row : rows) {
    items.clear();
    for (const QString &text : row)
      items.append(new QStandardItem(text));
    model.appendRow(items);
  }

  resultView->setModel(&model);
  resultView->verticalHeader()->hide();
  resultView->horizontalHeader()->setStretchLastSection(true);
}
