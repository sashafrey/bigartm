// Copyright 2011 Google Inc. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: nadavs@google.com <Nadav Samet>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/printer.h>

#include "rpcz/plugin/common/strutil.h"
#include "rpcz/plugin/cpp/cpp_helpers.h"
#include "rpcz/plugin/cpp/file_generator.h"
#include "rpcz/plugin/cpp/rpcz_cpp_service.h"

#ifdef _MSC_VER
#pragma warning( disable : 4244 )  // 'argument' : conversion from 'T1' to 'T2'
#endif

namespace rpcz {
namespace plugin {
namespace cpp {

using std::string;
using namespace ::google::protobuf;
using namespace google::protobuf::compiler::cpp;

FileGenerator::FileGenerator(const FileDescriptor* file,
                             const string& dllexport_decl)
    : file_(file), dllexport_decl_(dllexport_decl) {
  SplitStringUsing(file_->package(), ".", &package_parts_);
  for (int i = 0; i < file->service_count(); i++) {
    service_generators_.push_back(
      new ServiceGenerator(file->service(i), dllexport_decl));
  }
}

FileGenerator::~FileGenerator() {
  for (size_t i = 0; i < service_generators_.size(); i++) {
    delete service_generators_[i];
  }
}

void FileGenerator::GenerateHeader(io::Printer* printer) {
  string filename_identifier = FilenameIdentifier(file_->name());

  printer->Print(
    "// Generated by the protocol buffer compiler.  DO NOT EDIT!\n"
    "// source: $filename$\n"
    "\n"
    "#ifndef RPCZ_$filename_identifier$__INCLUDED\n"
    "#define RPCZ_$filename_identifier$__INCLUDED\n"
    "\n"
    "#include <string>\n"
    "#include <rpcz/service.hpp>\n"
    "\n"
    "namespace google {\n"
    "namespace protobuf {\n"
    "class ServiceDescriptor;\n"
    "class MethodDescriptor;\n"
    "}  // namespace protobuf\n"
    "}  // namespace google\n"
    "namespace rpcz {\n"
    "class rpc;\n"
    "class closure;\n"
    "class rpc_channel;\n"
    "}  //namesacpe rpcz\n"
    ,
    "filename", file_->name(),
    "filename_identifier", filename_identifier);

  for (int i = 0; i < file_->dependency_count(); i++) {
    printer->Print(
      "#include \"$dependency$.pb.h\"\n",
      "dependency", StripProto(file_->dependency(i)->name()));
  }

  printer->Print(
      "#include \"$dependency$.pb.h\"\n",
      "dependency", StripProto(file_->name()));

  GenerateNamespaceOpeners(printer);

  printer->Print(
    // Note that we don't put dllexport_decl on these because they are only
    // called by the .pb.cc file in which they are defined.
    "void rpcz_$assigndescriptorsname$();\n"
    "void rpcz_$shutdownfilename$();\n"
    "\n",
    "assigndescriptorsname", GlobalAssignDescriptorsName(file_->name()),
    "shutdownfilename", GlobalShutdownFileName(file_->name()));

  for (int i = 0; i < file_->service_count(); i++) {
    service_generators_[i]->GenerateDeclarations(printer);
  }

  GenerateNamespaceClosers(printer);
  printer->Print(
    "#endif  // RPCZ_$filename_identifier$__INCLUDED\n",
    "filename_identifier", filename_identifier);
}

void FileGenerator::GenerateSource(io::Printer* printer) {
  printer->Print(
    "// Generated by the protocol buffer compiler.  DO NOT EDIT!\n"
    "\n"
    "#include \"$basename$.rpcz.h\"\n"
    "#include \"$basename$.pb.h\"\n"
    "#include <google/protobuf/descriptor.h>\n"
    "#include <google/protobuf/stubs/once.h>\n"
    "#include <rpcz/rpcz.hpp>\n"
    "namespace {\n",
    "basename", StripProto(file_->name()));

  for (int i = 0; i < file_->service_count(); i++) {
    printer->Print(
      "const ::google::protobuf::ServiceDescriptor* $name$_descriptor_ = NULL;\n",
      "name", file_->service(i)->name());
  }
  printer->Print(
      "}  // anonymouse namespace\n");

  GenerateNamespaceOpeners(printer);
  GenerateBuildDescriptors(printer);

  for (int i = 0; i < file_->service_count(); i++) {
    if (i == 0) printer->Print("\n");
    printer->Print(kThickSeparator);
    printer->Print("\n");
    service_generators_[i]->GenerateImplementation(printer);
  }
  GenerateNamespaceClosers(printer);
}

void FileGenerator::GenerateNamespaceOpeners(io::Printer* printer) {
  if (package_parts_.size() > 0) printer->Print("\n");

  for (size_t i = 0; i < package_parts_.size(); i++) {
    printer->Print("namespace $part$ {\n",
                   "part", package_parts_[i]);
  }
}

void FileGenerator::GenerateNamespaceClosers(io::Printer* printer) {
  if (package_parts_.size() > 0) printer->Print("\n");

  for (int64 i = static_cast<int64>(package_parts_.size()) - 1; i >= 0; i--) {
    printer->Print("}  // namespace $part$\n",
                   "part", package_parts_[i]);
  }
}

void FileGenerator::GenerateBuildDescriptors(io::Printer* printer) {
  if (HasDescriptorMethods(file_)) {
    printer->Print(
      "\n"
      "void rpcz_$assigndescriptorsname$() {\n",
      "assigndescriptorsname", GlobalAssignDescriptorsName(file_->name()));
    printer->Indent();

    // Get the file's descriptor from the pool.
    printer->Print(
      "const ::google::protobuf::FileDescriptor* file =\n"
      "  ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(\n"
      "    \"$filename$\");\n"
      // Note that this GOOGLE_CHECK is necessary to prevent a warning about "file"
      // being unused when compiling an empty .proto file.
      "GOOGLE_CHECK(file != NULL);\n",
      "filename", file_->name());

    // Go through all the stuff defined in this file and generated code to
    // assign the global descriptor pointers based on the file descriptor.
    for (int i = 0; i < file_->service_count(); i++) {
      service_generators_[i]->GenerateDescriptorInitializer(printer, i);
    }

    printer->Outdent();
    printer->Print(
      "}\n"
      "\n");

    // ---------------------------------------------------------------

    // protobuf_AssignDescriptorsOnce():  The first time it is called, calls
    // AssignDescriptors().  All later times, waits for the first call to
    // complete and then returns.
    printer->Print(
      "namespace {\n"
      "\n"
      "GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);\n"
      "inline void protobuf_AssignDescriptorsOnce() {\n"
      "  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,\n"
      "                 &rpcz_$assigndescriptorsname$);\n"
      "}\n"
      "\n",
      "assigndescriptorsname", GlobalAssignDescriptorsName(file_->name()));

    // protobuf_RegisterTypes():  Calls
    // MessageFactory::InternalRegisterGeneratedType() for each message type.
    printer->Print(
      "void protobuf_RegisterTypes(const ::std::string&) {\n"
      "  protobuf_AssignDescriptorsOnce();\n");
    printer->Indent();

    printer->Outdent();
    printer->Print(
      "}\n"
      "\n"
      "}  // namespace\n");
  }

  // -----------------------------------------------------------------

  // ShutdownFile():  Deletes descriptors, default instances, etc. on shutdown.
  printer->Print(
    "\n"
    "void rpcz_$shutdownfilename$() {\n",
    "shutdownfilename", GlobalShutdownFileName(file_->name()));
  printer->Indent();

  printer->Outdent();
  printer->Print(
    "}\n");

  // -----------------------------------------------------------------

  // Now generate the AddDescriptors() function.
  printer->Print(
    "\n"
    "void rpcz_$adddescriptorsname$() {\n"
    // We don't need any special synchronization here because this code is
    // called at static init time before any threads exist.
    "  static bool already_here = false;\n"
    "  if (already_here) return;\n"
    "  already_here = true;\n"
    "  GOOGLE_PROTOBUF_VERIFY_VERSION;\n"
    "\n",
    "adddescriptorsname", GlobalAddDescriptorsName(file_->name()));
  printer->Indent();

  // Call the AddDescriptors() methods for all of our dependencies, to make
  // sure they get added first.
  for (int i = 0; i < file_->dependency_count(); i++) {
    const FileDescriptor* dependency = file_->dependency(i);
    // Print the namespace prefix for the dependency.
    vector<string> dependency_package_parts;
    SplitStringUsing(dependency->package(), ".", &dependency_package_parts);
    printer->Print("::");
    for (size_t i = 0; i < dependency_package_parts.size(); i++) {
      printer->Print("$name$::",
                     "name", dependency_package_parts[i]);
    }
    // Call its AddDescriptors function.
    printer->Print(
      "$name$();\n",
      "name", GlobalAddDescriptorsName(dependency->name()));
  }

  if (HasDescriptorMethods(file_)) {
    // Embed the descriptor.  We simply serialize the entire FileDescriptorProto
    // and embed it as a string literal, which is parsed and built into real
    // descriptors at initialization time.
    FileDescriptorProto file_proto;
    file_->CopyTo(&file_proto);
    string file_data;
    file_proto.SerializeToString(&file_data);

    printer->Print(
      "::google::protobuf::DescriptorPool::InternalAddGeneratedFile(");

    // Only write 40 bytes per line.
    static const int kBytesPerLine = 40;
    for (size_t i = 0; i < file_data.size(); i += kBytesPerLine) {
      printer->Print("\n  \"$data$\"",
        "data", EscapeTrigraphs(CEscape(file_data.substr(i, kBytesPerLine))));
    }
    printer->Print(
      ", $size$);\n",
      "size", SimpleItoa(file_data.size()));

    // Call MessageFactory::InternalRegisterGeneratedFile().
    printer->Print(
      "::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(\n"
      "  \"$filename$\", &protobuf_RegisterTypes);\n",
      "filename", file_->name());
  }

  printer->Print(
    "::google::protobuf::internal::OnShutdown(&rpcz_$shutdownfilename$);\n",
    "shutdownfilename", GlobalShutdownFileName(file_->name()));

  printer->Outdent();

  printer->Print("}\n");
}

}  // namespace cpp
}  // namespace plugin
}  // namespace rpcz
