set(SOURCES
    matrix_ops
    log_sigmoid
    tan_sigmoid
    file_reader
    ann
)

include_directories(${INCLUDE_DIRS})

# add_library(slog OBJECT ${SOURCES})
# add_library(slog_static STATIC $<TARGET_OBJECTS:soundex>)
# add_library(slog_shared SHARED $<TARGET_OBJECTS:soundex>)

add_library(ann_static STATIC ${SOURCES})
add_library(ann_shared SHARED ${SOURCES})

