@Library("jenkins-shared")_
rocmtest clang: rocmnode('vega', 'ubuntu') { cmake_build ->
    stage('Clang Debug') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('Clang Release') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, 
gcc: rocmnode('vega', 'ubuntu') { cmake_build ->
    stage('GCC Debug') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('GCC Release') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}
