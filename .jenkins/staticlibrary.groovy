#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCI =
{
    nodeDetails, jobName->

    def prj = new rocProject('hipGRAPH', 'Static Library PreCheckin')

    prj.paths.build_command = './install.sh -c --static'
    prj.compiler.compiler_name = 'hipcc'
    prj.compiler.compiler_path = '/opt/rocm/bin/hipcc'
    prj.libraryDependencies = ['rocBLAS', 'rocGRAPH', 'rocPRIM']
    prj.defaults.ccache = false

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runCompileCommand(platform, project,true)
    }

    def testCommand =
    {
        platform, project->

        def gfilter = "*checkin*"

        commonGroovy.runTestCommand(platform, project, gfilter)
    }

    def packageCommand =
    {
        platform, project->

        commonGroovy.runPackageCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 0')])],
                        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])],
                        "rocm-docker":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    Set standardJobNameSet = ["compute-rocm-dkms-no-npi", "compute-rocm-dkms-no-npi-hipclang", "rocm-docker"]

    def jobNameList = ["compute-rocm-dkms-no-npi":([ubuntu18:['gfx900'],centos7:['gfx908'],sles15sp1:['gfx906']]),
                       "compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900'],centos7:['gfx908'],sles15sp1:['gfx906']]),
                       "rocm-docker":([ubuntu18:['gfx900'],centos7:['gfx908'],sles15sp1:['gfx906']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    Set seenJobNames = []
    jobNameList.each
    {
        jobName, nodeDetails->
        seenJobNames.add(jobName)
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }

    // For url job names that are outside of the standardJobNameSet i.e. compute-rocm-dkms-no-npi-1901
    if(!seenJobNames.contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        runCI([ubuntu18:['gfx906']], urlJobName)
    }
}
