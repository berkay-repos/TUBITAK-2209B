#include <QtWidgets>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QWidget window;

    QLabel *queryLabel =
        new QLabel(QApplication::translate("nestedlayouts", "Query:"));
    QLineEdit *queryEdit = new QLineEdit();
    QTableView *resultView = new QTableView();

    QHBoxLayout *queryLayout = new QHBoxLayout();
    queryLayout->addWidget(queryLabel);
    queryLayout->addWidget(queryEdit);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->addLayout(queryLayout);
    mainLayout->addWidget(resultView);
    window.setLayout(mainLayout);

    // Set up the model and configure the view...

    QStandardItemModel model;
    model.setHorizontalHeaderLabels(
        {QApplication::translate("nestedlayouts", "Name"),
         QApplication::translate("nestedlayouts", "Office")});

    const QStringList rows[] = {
        QStringList{QStringLiteral("G          "), QStringLiteral("9.8  ")},
        QStringList{QStringLiteral("EPISODES   "), QStringLiteral("1000 ")},
        QStringList{QStringLiteral("vel_mps    "), QStringLiteral("20")},
        QStringList{QStringLiteral("n_episodes "), QStringLiteral("20000")},
        QStringList{QStringLiteral("LR         "), QStringLiteral("0.000")},
        QStringList{QStringLiteral("UpdateFreq "), QStringLiteral("4")},
        QStringList{QStringLiteral("action_time"), QStringLiteral("0.5  ")},
        QStringList{QStringLiteral("action_size"), QStringLiteral("15  ")},
        QStringList{QStringLiteral("eps_end    "), QStringLiteral("0.1  ")},
        QStringList{QStringLiteral("eps_decay  "), QStringLiteral("0.998")},
        QStringList{QStringLiteral("max_t      "), QStringLiteral("200  ")},
        QStringList{QStringLiteral("d_min     "), QStringLiteral(" 25 ")},
        QStringList{QStringLiteral("d_max     "), QStringLiteral(" 300  ")},
        QStringList{QStringLiteral("BATCH_SIZE"), QStringLiteral(" 128  ")},
        QStringList{QStringLiteral("rew_1     "), QStringLiteral(" True ")},
        QStringList{QStringLiteral("rew_2     "), QStringLiteral(" True ")},
        QStringList{QStringLiteral("blue_dom  "), QStringLiteral(" True ")},
        QStringList{QStringLiteral("red_dom   "), QStringLiteral(" True ")},
        QStringList{QStringLiteral("blue_oob  "), QStringLiteral(" False")},
        QStringList{QStringLiteral("col       "), QStringLiteral(" True ")},
        QStringList{QStringLiteral("red_oob    "), QStringLiteral("True ")}};
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
    window.setWindowTitle(
        QApplication::translate("nestedlayouts", "Nested layouts"));
    window.show();
    return app.exec();
}
