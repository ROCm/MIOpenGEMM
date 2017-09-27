
def rocmtestnode(variant, name, dockerfile, body) {
    def image = "miopengemm-${dockerfile}"
    def cmake_build = { compiler, flags ->
        def cmd = """
            rm -rf build
            mkdir build
            cd build
            CXX=${compiler} CXXFLAGS='-Werror' cmake ${flags} .. 
            CTEST_PARALLEL_LEVEL=4 dumb-init make -j32 check doc
        """
        echo cmd
        sh cmd
    }
    node(name) {
        stage("checkout ${variant}") {
            checkout scm
        }
        stage("image ${variant}") {
            try {
                docker.build("${image}", ". -f docker/${dockerfiler}.docker")
            } catch(Exception ex) {
                docker.build("${image}", "--no-cache . -f docker/${dockerfiler}.docker")

            }
        }
        withDockerContainer(image: image, args: '--device=/dev/kfd') {
            timeout(time: 1, unit: 'HOURS') {
                body(cmake_build)
            }
        }
    }
}
@NonCPS
def rocmtest(m) {
    def builders = [:]
    for(e in m) {
        def label = e.key;
        def action = e.value;
        builders[label] = {
            action(label)
        }
    }
    parallel builders
}

@NonCPS
def rocmnode(name, image, body) {
    def node_name = 'rocmtest || rocm'
    if(name == 'fiji') {
        node_name = 'rocmtest && fiji';
    } else if(name == 'vega') {
        node_name = 'rocmtest && vega';
    } else {
        node_name = name
    }
    return { label ->
        rocmtestnode(label, node_name, image, body)
    }
}

rocmtest clang: rocmnode('fiji', 'ubuntu') { cmake_build ->
    stage('Clang Debug') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('Clang Release') {
        cmake_build('clang++-3.8', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, gcc: rocmnode('fiji', 'ubuntu') { cmake_build ->
    stage('GCC Debug') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('GCC Release') {
        cmake_build('g++-5', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}, fedora: rocmnode('fiji', 'fedora') { cmake_build ->
    stage('Fedora GCC Debug') {
        cmake_build('g++', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=debug')
    }
    stage('Fedora GCC Release') {
        cmake_build('g++', '-DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release')
    }
}
