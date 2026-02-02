#include "gguf_loader.h"

#include <cstring>
#include <sstream>
#include <iostream>
#include <stdexcept>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace engine {

static constexpr uint32_t GGUF_VERSION_MIN = 2;
static constexpr uint32_t GGUF_VERSION_MAX = 3;
static constexpr uint64_t GGUF_ALIGNMENT = 32;

/* ================================================= */

static uint64_t align_up(uint64_t x, uint64_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

static uint32_t read_u32(const uint8_t*& p, const uint8_t* end) {
    if (p + 4 > end) throw std::runtime_error("GGUF truncated u32");
    uint32_t v;
    memcpy(&v, p, 4);
    p += 4;
    return v;
}

static uint64_t read_u64(const uint8_t*& p, const uint8_t* end) {
    if (p + 8 > end) throw std::runtime_error("GGUF truncated u64");
    uint64_t v;
    memcpy(&v, p, 8);
    p += 8;
    return v;
}

static std::string read_string(const uint8_t*& p, const uint8_t* end) {
    uint64_t len = read_u64(p, end);
    if (p + len > end) throw std::runtime_error("GGUF truncated string");
    std::string s((const char*)p, len);
    p += len;
    return s;
}

/* ================================================= */

enum class GgufValueType : uint32_t {
    UINT8=0, INT8=1, UINT16=2, INT16=3,
    UINT32=4, INT32=5, FLOAT32=6, BOOL=7,
    STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12
};

/* ---------- helpers ---------- */

static uint32_t read_value_as_u32(const uint8_t*& p,const uint8_t* end,GgufValueType t){
    if (t==GgufValueType::UINT32) return read_u32(p,end);
    if (t==GgufValueType::INT32){ int32_t v; memcpy(&v,p,4); p+=4; return (uint32_t)v; }
    if (t==GgufValueType::UINT64) return (uint32_t)read_u64(p,end);
    if (t==GgufValueType::INT64){ int64_t v; memcpy(&v,p,8); p+=8; return (uint32_t)v; }
    throw std::runtime_error("GGUF bad int value");
}

/* ---------- array readers ---------- */

static std::vector<std::string> read_array_string(const uint8_t*& p,const uint8_t* end){
    auto t=(GgufValueType)read_u32(p,end);
    if(t!=GgufValueType::STRING) throw std::runtime_error("GGUF expected string array");
    uint64_t n=read_u64(p,end);
    std::vector<std::string> out; out.reserve(n);
    for(uint64_t i=0;i<n;i++) out.push_back(read_string(p,end));
    return out;
}

static std::vector<float> read_array_f32(const uint8_t*& p,const uint8_t* end){
    auto t=(GgufValueType)read_u32(p,end);
    if(t!=GgufValueType::FLOAT32) throw std::runtime_error("GGUF expected f32 array");
    uint64_t n=read_u64(p,end);
    std::vector<float> out(n);
    memcpy(out.data(),p,n*4);
    p+=n*4;
    return out;
}

static std::vector<int32_t> read_array_i32(const uint8_t*& p,const uint8_t* end){
    auto t=(GgufValueType)read_u32(p,end);
    if(t!=GgufValueType::INT32) throw std::runtime_error("GGUF expected i32 array");
    uint64_t n=read_u64(p,end);
    std::vector<int32_t> out(n);
    memcpy(out.data(),p,n*4);
    p+=n*4;
    return out;
}

/* ---------- skip ---------- */

static void skip_value(const uint8_t*& p,const uint8_t* end,GgufValueType t);

static void skip_array(const uint8_t*& p,const uint8_t* end){
    auto et=(GgufValueType)read_u32(p,end);
    uint64_t n=read_u64(p,end);
    for(uint64_t i=0;i<n;i++) skip_value(p,end,et);
}

static void skip_value(const uint8_t*& p,const uint8_t* end,GgufValueType t){
    if(t==GgufValueType::ARRAY){ skip_array(p,end); return; }
    if(t==GgufValueType::STRING){ uint64_t l=read_u64(p,end); p+=l; return; }
    size_t sz=(t==GgufValueType::UINT8||t==GgufValueType::INT8||t==GgufValueType::BOOL)?1:
              (t==GgufValueType::UINT16||t==GgufValueType::INT16)?2:
              (t==GgufValueType::UINT32||t==GgufValueType::INT32||t==GgufValueType::FLOAT32)?4:8;
    p+=sz;
}

/* ================================================= */

uint64_t GgufTensorInfo::numel() const {
    uint64_t n=1;
    for(auto d:dims) n*=d;
    return n;
}

const void* GgufModel::tensor_ptr(const std::string& name) const {
    auto it=tensors_.find(name);
    if(it==tensors_.end()) return nullptr;
    return data_base_ + it->second.offset;
}

GgmlType GgufModel::tensor_type(const std::string& name) const {
    auto it=tensors_.find(name);
    if(it==tensors_.end()) return GgmlType::F32;
    return it->second.type;
}

const GgufTensorInfo* GgufModel::tensor_info(const std::string& name) const {
    auto it=tensors_.find(name);
    if(it==tensors_.end()) return nullptr;
    return &it->second;
}

std::string GgufModel::summary() const {
    std::ostringstream oss;
    oss<<"GGUF model: ctx="<<context_length_
       <<" emb="<<embedding_dim_
       <<" layers="<<n_layers_
       <<" tensors="<<tensors_.size()
       <<" file_size="<<file_size_;
    return oss.str();
}

/* ================================================= */

void GgufLoader::validate_magic(const char magic[4]) {
    if (!(magic[0] == 'G' &&
          magic[1] == 'G' &&
          magic[2] == 'U' &&
          magic[3] == 'F')) {
        throw std::runtime_error("GGUF: invalid magic");
    }
}


/* ================================================= */

GgufModel GgufLoader::load(const std::string& path){

#ifndef __linux__
    throw std::runtime_error("GGUF loader requires linux mmap");
#else

    int fd=open(path.c_str(),O_RDONLY);
    if(fd<0) throw std::runtime_error("GGUF open failed");

    struct stat st{};
    fstat(fd,&st);

    void* base=mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);

    if(base==MAP_FAILED) throw std::runtime_error("GGUF mmap failed");

    GgufModel model;
    model.file_base_=base;
    model.file_size_=st.st_size;

    const uint8_t* p=(const uint8_t*)base;
    const uint8_t* end=p+model.file_size_;

    char magic[4];
    memcpy(magic,p,4); p+=4;
    validate_magic(magic);

    uint32_t version=read_u32(p,end);
    if(version<GGUF_VERSION_MIN||version>GGUF_VERSION_MAX)
        throw std::runtime_error("GGUF unsupported version");

    uint64_t n_tensors=read_u64(p,end);
    uint64_t n_kv=read_u64(p,end);

    /* ================= KV ================= */

    for(uint64_t i=0;i<n_kv;i++){
        std::string key=read_string(p,end);
        auto t=(GgufValueType)read_u32(p,end);

        if(key=="llama.context_length"||key=="n_ctx"){
            model.context_length_=read_value_as_u32(p,end,t); continue;
        }
        if(key=="llama.embedding_length"||key=="n_embd"){
            model.embedding_dim_=read_value_as_u32(p,end,t); continue;
        }
        if(key=="llama.block_count"||key=="n_layer"){
            model.n_layers_=read_value_as_u32(p,end,t); continue;
        }

        /* ---- tokenizer ---- */

        if(key=="tokenizer.ggml.tokens"){
            model.tokenizer_tokens_=read_array_string(p,end); continue;
        }
        if(key=="tokenizer.ggml.token_scores"){
            model.tokenizer_scores_=read_array_f32(p,end); continue;
        }
        if(key=="tokenizer.ggml.token_types"){
            model.tokenizer_types_=read_array_i32(p,end); continue;
        }

        if(key=="tokenizer.ggml.bos_token_id"){
            model.bos_id_=(int32_t)read_value_as_u32(p,end,t); continue;
        }
        if(key=="tokenizer.ggml.eos_token_id"){
            model.eos_id_=(int32_t)read_value_as_u32(p,end,t); continue;
        }
        if(key=="tokenizer.ggml.unk_token_id"){
            model.unk_id_=(int32_t)read_value_as_u32(p,end,t); continue;
        }

        skip_value(p,end,t);
    }

    /* ================= TENSORS ================= */

    for(uint64_t i=0;i<n_tensors;i++){
        GgufTensorInfo info;
        info.name=read_string(p,end);
        info.n_dims=read_u32(p,end);
        info.dims.resize(info.n_dims);
        for(uint32_t d=0;d<info.n_dims;d++)
            info.dims[d]=read_u64(p,end);
        info.type=(GgmlType)read_u32(p,end);
        info.offset=read_u64(p,end);
        model.tensors_.emplace(info.name,std::move(info));
    }

    uint64_t cur=(uint64_t)(p-(const uint8_t*)base);
    model.data_offset_=align_up(cur,GGUF_ALIGNMENT);
    model.data_base_=(const uint8_t*)base + model.data_offset_;

    return model;

#endif
}

} // namespace engine
