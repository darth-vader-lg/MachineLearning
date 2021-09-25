using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Text;

namespace Common.Examples
{
   /// <summary>
   /// Test file
   /// </summary>
   public class ExampleData
   {
      #region Fields
      /// <summary>
      /// Optional custom builder
      /// </summary>
      private readonly Builder builder;
      /// <summary>
      /// Define if it's a folder
      /// </summary>
      private readonly bool isFolder;
      /// <summary>
      /// Path relative to the root
      /// </summary>
      private readonly string path;
      /// <summary>
      /// Root of the path on disk
      /// </summary>
      private readonly string root;
      /// <summary>
      /// Url for download
      /// </summary>
      private readonly string url;
      #endregion
      #region Properties
      /// <summary>
      /// The full path of the destination
      /// </summary>
      public string FullPath => Path.GetFullPath(Path.Combine(root, path));
      #endregion
      #region Delegates
      /// <summary>
      /// Custom creator delegate
      /// </summary>
      /// <param name="fullPath">Full path of the object to create</param>
      public delegate void Builder(string fullPath);
      #endregion
      #region Methods
      /// <summary>
      /// Constructor
      /// </summary>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="url">Url for download</param>
      /// <param name="isFolder">Interpreted as folder</param>
      /// <param name="builder">Optional custom builder</param>
      private ExampleData(string root, string path, string url, bool isFolder, Builder builder = null)
      {
         this.builder = builder;
         this.root = root;
         this.path = path;
         this.url = url;
         this.isFolder = isFolder;
      }
      /// <summary>
      /// Create an instance of a test data file
      /// </summary>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="builder">Optional custom builder</param>
      public static ExampleData File(string root, string path, Builder builder = null) => new(root, path, null, false, builder);
      /// <summary>
      /// Create an instance of a test data file
      /// </summary>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="url">Url for download</param>
      public static ExampleData File(string root, string path, string url) => new(root, path, url, false, null);
      /// <summary>
      /// Create an instance of a test data folder
      /// </summary>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="builder">Optional custom builder</param>
      public static ExampleData Folder(string root, string path, Builder builder = null) => new(root, path, null, true, builder);
      /// <summary>
      /// Create an instance of a test data folder
      /// </summary>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="url">Url for download</param>
      public static ExampleData Folder(string root, string path, string url) => new(root, path, url, true, null);
      /// <summary>
      /// Download/extract and get the path on disk
      /// </summary>
      /// <returns>The path of the test object</returns>
      public string Get()
      {
         // Check for existence
         var fullPath = Path.GetFullPath(Path.Combine(root, path));
         if ((isFolder && Directory.Exists(fullPath)) || (!isFolder && System.IO.File.Exists(fullPath)))
            return fullPath;
         // Custom build of test data
         if (builder != null)
            builder(fullPath);
         else {
            var extract = isFolder || !string.IsNullOrEmpty(Path.GetDirectoryName(path));
            if (extract && Path.GetExtension(url).ToLower() == ".gz") {
               var tmpGZip = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
               var tmpTar = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
               try {
                  var webClient = new WebClient();
                  webClient.DownloadFile(url, tmpGZip);
                  GZip.Decompress(System.IO.File.OpenRead(tmpGZip), System.IO.File.Create(tmpTar), true);
                  using var tarStream = System.IO.File.OpenRead(tmpTar);
                  using var tar = TarArchive.CreateInputTarArchive(tarStream, Encoding.UTF8);
                  if (!Directory.Exists(root))
                     Directory.CreateDirectory(root);
                  tar.ExtractContents(root);
               }
               finally {
                  System.IO.File.Delete(tmpGZip);
                  System.IO.File.Delete(tmpTar);
               }
            }
            else if (extract && Path.GetExtension(url).ToLower() == ".zip") {
               var tmpZip = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
               try {
                  var webClient = new WebClient();
                  webClient.DownloadFile(url, tmpZip);
                  if (!Directory.Exists(root))
                     Directory.CreateDirectory(root);
                  ZipFile.ExtractToDirectory(tmpZip, root, true);
               }
               finally {
                  System.IO.File.Delete(tmpZip);
               }
            }
            else {
               if (!Directory.Exists(root))
                  Directory.CreateDirectory(root);
               var webClient = new WebClient();
               webClient.DownloadFile(url, fullPath);
            }
         }
         return fullPath;
      }
      #endregion
   }
}
