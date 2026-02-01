#include "model/tokenizer.h"
#include "model/gguf_loader.h"

#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <iostream>

namespace engine {

bool SimpleTokenizer::load_from_gguf(const std::string& model_path) {
    // TODO FASE 3.5: Extrair vocabulário do GGUF
    // Por enquanto, vocabulário mínimo hard-coded

    vocab_.resize(32000);  // Tamanho típico do Mistral

    // Tokens especiais
    vocab_[0] = "<pad>";
    vocab_[1] = "<s>";
    vocab_[2] = "</s>";
    vocab_[3] = "<unk>";

    // Caracteres ASCII básicos (fallback)
    for (int i = 32; i < 127; ++i) {
        vocab_[i] = std::string(1, static_cast<char>(i));
    }

    // Palavras comuns (stub)
    vocab_[100] = " hello";
    vocab_[101] = " world";
    vocab_[102] = " the";
    vocab_[103] = " is";

    return true;
}

std::vector<int32_t> SimpleTokenizer::encode(const std::string& text) const {
    // FASE 3: Tokenização simplificada por whitespace
    // FASE 4: Integrar SentencePiece real

    if (vocab_.empty()) {
        return encode_whitespace(text);
    }

    // Tokenização básica: BOS + palavras + EOS
    std::vector<int32_t> tokens;
    tokens.push_back(bos_token_);

    // Split por espaço
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        // Procura no vocabulário (busca linear - ineficiente!)
        bool found = false;
        for (size_t i = 0; i < vocab_.size(); ++i) {
            if (vocab_[i] == " " + word || vocab_[i] == word) {
                tokens.push_back(static_cast<int32_t>(i));
                found = true;
                break;
            }
        }

        if (!found) {
            // Token desconhecido → usar IDs dos caracteres
            for (char c : word) {
                if (c >= 32 && c < 127) {
                    tokens.push_back(static_cast<int32_t>(c));
                }
            }
        }
    }

    tokens.push_back(eos_token_);
    return tokens;
}

std::string SimpleTokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;

    std::cout << "[tokenizer] decoding " << tokens.size() << " tokens\n";

    for (int32_t token_id : tokens) {
        // Ignora tokens especiais
        if (token_id == bos_token_ || token_id == eos_token_ || token_id == pad_token_) {
            std::cout << "[tokenizer] skipping special token: " << token_id << "\n";
            continue;
        }

        // Verifica bounds
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_.size()) {
            std::cout << "[tokenizer] WARNING: token_id " << token_id << " out of range\n";
            result += "<unk>";
            continue;
        }

        std::cout << "[tokenizer] token " << token_id << " -> '" << vocab_[token_id] << "'\n";
        result += vocab_[token_id];
    }

    return result;
}

std::vector<int32_t> SimpleTokenizer::encode_whitespace(const std::string& text) const {
    // Fallback ultra-simples: cada palavra → um ID único
    std::vector<int32_t> tokens;
    tokens.push_back(bos_token_);

    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        // Hash simples da palavra
        uint32_t hash = 0;
        for (char c : word) {
            hash = hash * 31 + static_cast<uint32_t>(c);
        }

        // Mapeia para faixa do vocabulário
        int32_t token_id = 100 + (hash % 30000);
        tokens.push_back(token_id);
    }

    tokens.push_back(eos_token_);
    return tokens;
}

} // namespace engine