add_executable(webcam main.cpp)
target_link_libraries(webcam PRIVATE ${OpenCV_LIBS})

add_executable(vo vo.cpp)
target_link_libraries(vo PRIVATE ${OpenCV_LIBS})