//
// Created by 张宇 on 2025/3/27.
//

#ifndef WAV2LIP_CPP_PROCESSAUDUUSEPY_HPP
#define WAV2LIP_CPP_PROCESSAUDUUSEPY_HPP

#include <vector>
#include <iostream>
#include <python3.9/Python.h>

PyObject *vector_to_pylist(const std::vector<double> &vec) {
    PyObject *pylist = PyList_New(vec.size());
    if (!pylist) return nullptr;

    for (size_t i = 0; i < vec.size(); ++i) {
        PyObject *pyfloat = PyFloat_FromDouble(vec[i]);
        if (!pyfloat) {
            Py_DECREF(pylist);
            return nullptr;
        }
        PyList_SetItem(pylist, i, pyfloat);
    }

    return pylist;
}

std::vector<std::vector<std::vector<float>>> callPythonFunction(const std::vector<double> &input) {
    Py_SetPythonHome((wchar_t *) L"/opt/miniconda/envs/py39"); // *******3
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/Users/zhangyu/CLionProjects/wav2lip_cpp')");
    PyObject *pName = PyUnicode_DecodeFSDefault("audioProcPY");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // 获取Python函数
        PyObject *pFunc = PyObject_GetAttrString(pModule, "audioProc");
        if (pFunc && PyCallable_Check(pFunc)) {
            // 创建Python列表来传递std::vector<double>
            auto pylist = vector_to_pylist(input);
            PyObject* pArgs = PyTuple_New(1);
            PyTuple_SetItem(pArgs, 0, pylist); // 设置元组项，并转移引用所有权

            // 调用Python函数
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != nullptr) {
                // 解析返回的Python对象为std::vector<std::vector<std::vector<double>>>
                std::vector<std::vector<std::vector<float>>> result;
                if (PyList_Check(pValue)) {
                    for (Py_ssize_t i = 0; i < PyList_Size(pValue); ++i) {
                        PyObject *pList2 = PyList_GetItem(pValue, i);
                        std::vector<std::vector<float>> vec2;
                        if (PyList_Check(pList2)) {
                            for (Py_ssize_t j = 0; j < PyList_Size(pList2); ++j) {
                                PyObject *pList3 = PyList_GetItem(pList2, j);
                                std::vector<float> vec3;
                                if (PyList_Check(pList3)) {
                                    for (Py_ssize_t k = 0; k < PyList_Size(pList3); ++k) {
                                        PyObject *pDouble = PyList_GetItem(pList3, k);
                                        vec3.push_back(PyFloat_AsDouble(pDouble));
                                    }
                                }
                                vec2.push_back(vec3);
                            }
                        }
                        result.push_back(vec2);
                    }
                }
                Py_DECREF(pValue);
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_Finalize();
                return result;
            } else {
                PyErr_Print();
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                Py_Finalize();
                throw std::runtime_error("Call to Python function failed");
            }
        } else {
            if (PyErr_Occurred()) PyErr_Print();
            Py_DECREF(pModule);
            Py_Finalize();
            throw std::runtime_error("Cannot find function 'your python function'");
        }
    } else {
        PyErr_Print();
        Py_Finalize();
        throw std::runtime_error("Failed to load 'your python module'");
    }
}


#endif //WAV2LIP_CPP_PROCESSAUDUUSEPY_HPP
