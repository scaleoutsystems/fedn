class ModelDto:
    Bucket = None
    Name = None
    LastModified = None
    ETag = None
    Size = None
    ContentType = None
    Link = None

    def __init__(self, bucket, name, last_modified, etag, size, content_type, link):
        self.Bucket = bucket
        self.Name = name
        self.LastModified = last_modified
        self.ETag = etag
        self.Size = size
        self.ContentType = content_type
        self.Link = link
