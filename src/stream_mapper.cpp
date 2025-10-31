#include "stream_mapper.h"
#include "logger.h"
#include <cstring>
#include <cctype>
#include <algorithm>

// Parse media type character (v, a, s, d, t) to AVMediaType
static AVMediaType parse_media_type(char c)
{
    switch (c)
    {
    case 'v':
    case 'V':
        return AVMEDIA_TYPE_VIDEO;
    case 'a':
        return AVMEDIA_TYPE_AUDIO;
    case 's':
        return AVMEDIA_TYPE_SUBTITLE;
    case 'd':
        return AVMEDIA_TYPE_DATA;
    case 't':
        return AVMEDIA_TYPE_ATTACHMENT;
    default:
        return AVMEDIA_TYPE_UNKNOWN;
    }
}

// Parse a single -map argument string into a StreamMapSpec
// Supports FFmpeg syntax:
// - "0:0" = input 0, stream 0
// - "0:v" = input 0, all video streams
// - "0:v:0" = input 0, first video stream
// - "0:a?" = input 0, audio streams (optional, don't error if missing)
// - "-0:1" = negative map (exclude stream)
// - "0:m:language:eng" = streams with metadata language=eng
// - "0:m:title:foo" = streams with metadata title=foo
bool parse_map_spec(const std::string &map_arg, StreamMapSpec &spec)
{
    spec = StreamMapSpec(); // Reset to defaults
    spec.raw_specifier = map_arg;

    const char *str = map_arg.c_str();
    const char *p = str;

    // Check for negative map (leading '-')
    if (*p == '-')
    {
        spec.is_negative = true;
        p++;
    }

    // Parse input file index (required)
    if (!std::isdigit(*p))
    {
        LOG_ERROR("Invalid map specifier '%s': missing input file index", map_arg.c_str());
        return false;
    }

    char *endptr;
    spec.input_file_index = std::strtol(p, &endptr, 10);
    p = endptr;

    // If no colon, it means "map all streams from this input"
    if (*p == '\0' || *p == '?')
    {
        if (*p == '?')
        {
            spec.is_optional = true;
        }
        // Map all streams from input
        return true;
    }

    // Parse stream specifier after ':'
    if (*p != ':')
    {
        LOG_ERROR("Invalid map specifier '%s': expected ':' after input index", map_arg.c_str());
        return false;
    }
    p++; // Skip ':'

    // Check for metadata filter (m:key:value or m:key:value:index)
    if (*p == 'm' && *(p + 1) == ':')
    {
        p += 2; // Skip "m:"

        // Parse metadata key
        const char *key_start = p;
        while (*p && *p != ':' && *p != '?')
            p++;

        if (*p != ':')
        {
            LOG_ERROR("Invalid metadata filter in '%s': expected 'key:value' after 'm:'", map_arg.c_str());
            return false;
        }

        spec.metadata_key = std::string(key_start, p - key_start);
        p++; // Skip ':'

        // Parse metadata value
        const char *value_start = p;
        while (*p && *p != ':' && *p != '?')
            p++;

        spec.metadata_value = std::string(value_start, p - value_start);
        spec.has_metadata_filter = true;

        // Check for index after metadata value (e.g., "m:language:eng:0")
        if (*p == ':')
        {
            p++;
            if (std::isdigit(*p))
            {
                spec.stream_type_index = std::strtol(p, &endptr, 10);
                p = endptr;
            }
        }
    }
    // Check for media type character (v, a, s, d, t)
    else if ((*p == 'v' || *p == 'V' || *p == 'a' || *p == 's' || *p == 'd' || *p == 't') &&
             (!std::isalnum(*(p + 1)) || *(p + 1) == ':' || *(p + 1) == '?'))
    {
        spec.stream_type = parse_media_type(*p);
        p++;

        // Check for type index after media type (e.g., "v:0")
        if (*p == ':')
        {
            p++;
            if (std::isdigit(*p))
            {
                spec.stream_type_index = std::strtol(p, &endptr, 10);
                p = endptr;
            }
        }
    }
    else if (std::isdigit(*p))
    {
        // Direct stream index (e.g., "0:1" means stream index 1)
        spec.stream_index = std::strtol(p, &endptr, 10);
        p = endptr;
    }

    // Check for optional marker '?'
    if (*p == '?')
    {
        spec.is_optional = true;
        p++;
    }

    // Ensure no trailing garbage
    if (*p != '\0')
    {
        LOG_WARN("Map specifier '%s' has trailing characters: '%s'", map_arg.c_str(), p);
    }

    return true;
}

// Check if a stream matches a stream specifier
bool stream_matches_spec(
    const AVStream *stream,
    int stream_index,
    const StreamMapSpec &spec)
{
    // Check metadata filter first (e.g., "0:m:language:eng")
    if (spec.has_metadata_filter)
    {
        AVDictionaryEntry *tag = av_dict_get(stream->metadata, spec.metadata_key.c_str(), nullptr, 0);
        if (!tag)
        {
            // No metadata with this key
            return false;
        }

        // Check if value matches
        if (spec.metadata_value != tag->value)
        {
            return false;
        }

        // Metadata matches! Continue with other filters...
    }

    // Check specific stream index
    if (spec.stream_index >= 0 && spec.stream_index != stream_index)
    {
        return false;
    }

    // Check media type filter
    if (spec.stream_type != AVMEDIA_TYPE_UNKNOWN)
    {
        if (stream->codecpar->codec_type != spec.stream_type)
        {
            return false;
        }

        // Check stream type index (e.g., "v:0" means first video stream)
        if (spec.stream_type_index >= 0)
        {
            // Count streams of this type before this stream
            // This requires access to the format context, so we'll handle this in the caller
            // For now, return true and let the caller handle type index filtering
        }
    }

    return true;
}

// Apply default FFmpeg stream selection (when no -map is specified)
void apply_default_mapping(
    const std::vector<InputContext> &inputs,
    const PipelineConfig &cfg,
    std::vector<MappedStream> &mapped_streams)
{
    // FFmpeg default behavior: select one stream of each type from the first input
    // that has such streams, unless disabled with -vn, -an, etc.

    if (inputs.empty())
    {
        return;
    }

    // Find best stream of each type from first input
    const InputContext &first_input = inputs[0];

    // Video stream (unless -vn)
    if (!cfg.disableVideo && first_input.vstream >= 0)
    {
        MappedStream ms;
        ms.input_index = 0;
        ms.input_stream_index = first_input.vstream;
        ms.type = AVMEDIA_TYPE_VIDEO;
        ms.requires_processing = true; // Video needs encoding
        ms.copy_stream = false;
        mapped_streams.push_back(ms);
        LOG_DEBUG("Default mapping: input 0 stream %d (video)", first_input.vstream);
    }

    // Audio stream (unless -an)
    if (!cfg.disableAudio && first_input.primary_audio_stream >= 0)
    {
        MappedStream ms;
        ms.input_index = 0;
        ms.input_stream_index = first_input.primary_audio_stream;
        ms.type = AVMEDIA_TYPE_AUDIO;
        ms.requires_processing = true; // Audio needs encoding
        ms.copy_stream = false;
        mapped_streams.push_back(ms);
        LOG_DEBUG("Default mapping: input 0 stream %d (audio)", first_input.primary_audio_stream);
    }

    // Note: FFmpeg doesn't map subtitle/data streams by default
}

// Resolve stream mapping from parsed specs and input contexts
std::vector<MappedStream> resolve_stream_mapping(
    const std::vector<InputContext> &inputs,
    const PipelineConfig &cfg)
{
    std::vector<MappedStream> mapped_streams;

    // If no explicit -map arguments, use default mapping
    if (cfg.streamMaps.empty())
    {
        apply_default_mapping(inputs, cfg, mapped_streams);
        return mapped_streams;
    }

    // Parse all map specs
    std::vector<StreamMapSpec> specs;
    for (const std::string &map_arg : cfg.streamMaps)
    {
        StreamMapSpec spec;
        if (parse_map_spec(map_arg, spec))
        {
            specs.push_back(spec);
        }
        else
        {
            LOG_ERROR("Failed to parse -map argument: %s", map_arg.c_str());
        }
    }

    // Apply positive maps first
    for (const StreamMapSpec &spec : specs)
    {
        if (spec.is_negative)
        {
            continue; // Handle negative maps in second pass
        }

        // Validate input file index
        if (spec.input_file_index < 0 || spec.input_file_index >= (int)inputs.size())
        {
            if (!spec.is_optional)
            {
                LOG_ERROR("Invalid input file index in -map: %d (have %d inputs)",
                          spec.input_file_index, (int)inputs.size());
            }
            continue;
        }

        const InputContext &input = inputs[spec.input_file_index];
        int type_counter[AVMEDIA_TYPE_NB] = {0}; // Count streams by type for type index matching

        // Map all streams from this input
        if (spec.stream_index < 0 && spec.stream_type == AVMEDIA_TYPE_UNKNOWN)
        {
            // Map all streams from this input
            for (unsigned i = 0; i < input.fmt->nb_streams; i++)
            {
                AVStream *st = input.fmt->streams[i];
                MappedStream ms;
                ms.input_index = spec.input_file_index;
                ms.input_stream_index = i;
                ms.type = st->codecpar->codec_type;

                // Determine if processing is needed
                if (ms.type == AVMEDIA_TYPE_VIDEO || ms.type == AVMEDIA_TYPE_AUDIO)
                {
                    ms.requires_processing = true;
                    ms.copy_stream = false;
                }
                else
                {
                    ms.requires_processing = false;
                    ms.copy_stream = true; // Copy subtitles, data, etc.
                }

                mapped_streams.push_back(ms);
                LOG_DEBUG("Mapped input %d stream %d (%s)", spec.input_file_index, i,
                          av_get_media_type_string(ms.type));
            }
            continue;
        }

        // Map specific stream(s) matching the specifier
        bool found_match = false;
        for (unsigned i = 0; i < input.fmt->nb_streams; i++)
        {
            AVStream *st = input.fmt->streams[i];

            // Check if stream matches spec
            if (!stream_matches_spec(st, i, spec))
            {
                if (st->codecpar->codec_type == spec.stream_type && spec.stream_type != AVMEDIA_TYPE_UNKNOWN)
                {
                    type_counter[st->codecpar->codec_type]++;
                }
                continue;
            }

            // Check stream type index (e.g., "v:0" means first video stream)
            if (spec.stream_type_index >= 0)
            {
                if (type_counter[st->codecpar->codec_type] != spec.stream_type_index)
                {
                    type_counter[st->codecpar->codec_type]++;
                    continue;
                }
            }

            // Stream matches! Add to output
            MappedStream ms;
            ms.input_index = spec.input_file_index;
            ms.input_stream_index = i;
            ms.type = st->codecpar->codec_type;

            // Determine if processing is needed
            if (ms.type == AVMEDIA_TYPE_VIDEO || ms.type == AVMEDIA_TYPE_AUDIO)
            {
                ms.requires_processing = true;
                ms.copy_stream = false;
            }
            else
            {
                ms.requires_processing = false;
                ms.copy_stream = true; // Copy subtitles, data, etc.
            }

            mapped_streams.push_back(ms);
            found_match = true;
            LOG_DEBUG("Mapped input %d stream %d (%s) via spec '%s'",
                      spec.input_file_index, i,
                      av_get_media_type_string(ms.type),
                      spec.raw_specifier.c_str());

            // If mapping specific stream index, stop after first match
            if (spec.stream_index >= 0 || spec.stream_type_index >= 0)
            {
                break;
            }

            if (st->codecpar->codec_type == spec.stream_type && spec.stream_type != AVMEDIA_TYPE_UNKNOWN)
            {
                type_counter[st->codecpar->codec_type]++;
            }
        }

        if (!found_match && !spec.is_optional)
        {
            LOG_WARN("No matching streams for -map '%s'", spec.raw_specifier.c_str());
        }
    }

    // Apply negative maps (remove streams matching negative specs)
    for (const StreamMapSpec &spec : specs)
    {
        if (!spec.is_negative)
        {
            continue;
        }

        // Validate input file index
        if (spec.input_file_index < 0 || spec.input_file_index >= (int)inputs.size())
        {
            continue;
        }

        const InputContext &input = inputs[spec.input_file_index];

        // Remove streams matching this spec
        mapped_streams.erase(
            std::remove_if(mapped_streams.begin(), mapped_streams.end(),
                           [&](const MappedStream &ms) {
                               if (ms.input_index != spec.input_file_index)
                               {
                                   return false;
                               }

                               AVStream *st = input.fmt->streams[ms.input_stream_index];
                               return stream_matches_spec(st, ms.input_stream_index, spec);
                           }),
            mapped_streams.end());
    }

    // Apply stream disable flags (-vn, -an, -sn, -dn)
    if (cfg.disableVideo || cfg.disableAudio || cfg.disableSubtitle || cfg.disableData)
    {
        mapped_streams.erase(
            std::remove_if(mapped_streams.begin(), mapped_streams.end(),
                           [&](const MappedStream &ms) {
                               if (cfg.disableVideo && ms.type == AVMEDIA_TYPE_VIDEO)
                                   return true;
                               if (cfg.disableAudio && ms.type == AVMEDIA_TYPE_AUDIO)
                                   return true;
                               if (cfg.disableSubtitle && ms.type == AVMEDIA_TYPE_SUBTITLE)
                                   return true;
                               if (cfg.disableData && ms.type == AVMEDIA_TYPE_DATA)
                                   return true;
                               return false;
                           }),
            mapped_streams.end());
    }

    LOG_INFO("Stream mapping resolved: %d output streams", (int)mapped_streams.size());
    return mapped_streams;
}
