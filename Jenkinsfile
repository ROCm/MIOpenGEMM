pipeline {
    agent none
    environment{
        image = "ubuntu"
    }
    stages {
        stage("build and deploy for clang and gcc") {
            parallel {
                stage("clang") {
                    agent{  label rocmnode("vega") }
                    stages {
                        stage("clang Debug") {
                            steps {
                                cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
                            }
                        }
                        stage("Clang Release") {
                            steps {
                                cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                            }
                        }
                    }
                }

                stage("gcc") {
                    agent{  label rocmnode("vega") }
                    stages {
                        stage("GCC Debug") {
                            steps {
                                echo "running parllel-2"
                                cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
                            }
                        }
                        stage("GCC Release") {
                             steps {
                                echo "running parallel-2.1"
                                cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
                            }
                        }
                    }
                }
            }
        }
    }
}