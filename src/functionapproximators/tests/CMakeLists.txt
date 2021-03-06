add_executable(testTrainingCompareCppPython testTrainingCompareCppPython.cpp)
target_link_libraries(testTrainingCompareCppPython functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testTrainingCompareCppPython DESTINATION bin_test)

add_executable(testFunctionApproximatorCloning testFunctionApproximatorCloning.cpp)
target_link_libraries(testFunctionApproximatorCloning functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testFunctionApproximatorCloning DESTINATION bin_test)

add_executable(testFunctionApproximatorRealtime testFunctionApproximatorRealtime.cpp)
target_link_libraries(testFunctionApproximatorRealtime functionapproximators ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})
install(TARGETS testFunctionApproximatorRealtime DESTINATION bin_test)

add_executable(testFunctionApproximatorSerialization testFunctionApproximatorSerialization.cpp)
target_link_libraries(testFunctionApproximatorSerialization functionapproximators ${LWPR_LIBRARIES}  ${Boost_LIBRARIES})
install(TARGETS testFunctionApproximatorSerialization DESTINATION bin_test)

add_executable(testModelParameterValues testModelParameterValues.cpp)
target_link_libraries(testModelParameterValues functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testModelParameterValues  DESTINATION bin_test)

add_executable(testToUnifiedModel testToUnifiedModel.cpp)
target_link_libraries(testToUnifiedModel functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testToUnifiedModel  DESTINATION bin_test)



################################################################################
# LWR
add_executable(testMetaParametersLWR testMetaParametersLWR.cpp)
target_link_libraries(testMetaParametersLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testMetaParametersLWR  DESTINATION bin_test)

add_executable(testPerturbModelParameterLWR testPerturbModelParameterLWR.cpp)
target_link_libraries(testPerturbModelParameterLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testPerturbModelParameterLWR  DESTINATION bin_test)

add_executable(testBasisFunctionsLWR testBasisFunctionsLWR.cpp)
target_link_libraries(testBasisFunctionsLWR functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testBasisFunctionsLWR  DESTINATION bin_test)


################################################################################
# GMR
add_executable(testSaveGMMMatrix testSaveGMMMatrix.cpp)
target_link_libraries(testSaveGMMMatrix functionapproximators ${LWPR_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS testSaveGMMMatrix  DESTINATION bin_test)

add_executable(testMultiDimOutput testMultiDimOutput.cpp)
target_link_libraries(testMultiDimOutput functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS testMultiDimOutput  DESTINATION bin_test)

add_executable(testIncrementalTrainingGMR testIncrementalTrainingGMR.cpp)
target_link_libraries(testIncrementalTrainingGMR functionapproximators ${Boost_LIBRARIES} ${LWPR_LIBRARIES})
install(TARGETS testIncrementalTrainingGMR  DESTINATION bin_test)

