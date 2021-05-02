HEADERS       = ../connection.h
SOURCES       = main.cpp
QT           += sql widgets
requires(qtConfig(tableview))

# install
target.path = $$[QT_INSTALL_EXAMPLES]/sql/tablemodel
INSTALLS += target
