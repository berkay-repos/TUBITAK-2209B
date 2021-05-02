QT += widgets
CONFIG += c++0x
requires(qtConfig(tableview))

HEADERS     = slidersgroup.h \
              tablegroup.h \
              window.h
SOURCES     = main.cpp \
              slidersgroup.cpp \
              tablegroup.cpp \
              window.cpp

# install
target.path = build/
INSTALLS += target
