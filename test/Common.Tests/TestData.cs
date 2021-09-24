using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Text;
using Xunit;
using IOFile = System.IO.File;
using IOPath = System.IO.Path;

namespace Common.Tests
{
   /// <summary>
   /// Test file
   /// </summary>
   public class TestData
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
      public string FullPath => IOPath.GetFullPath(IOPath.Combine(root, path));
      /// <summary>
      /// Mnemonic name of the object
      /// </summary>
      public string Name { get; }
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
      /// <param name="name">Name of the object</param>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="url">Url for download</param>
      /// <param name="isFolder">Interpreted as folder</param>
      /// <param name="builder">Optional custom builder</param>
      private TestData(string name, string root, string path, string url, bool isFolder, Builder builder = null)
      {
         Name = name;
         this.builder = builder;
         this.root = root;
         this.path = path;
         this.url = url;
         this.isFolder = isFolder;
      }
      /// <summary>
      /// Create an instance of a test data file
      /// </summary>
      /// <param name="name">Name of the object</param>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="builder">Optional custom builder</param>
      public static TestData File(string name, string root, string path, Builder builder = null) => new(name, root, path, null, false, builder);
      /// <summary>
      /// Create an instance of a test data file
      /// </summary>
      /// <param name="name">Name of the object</param>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="url">Url for download</param>
      public static TestData File(string name, string root, string path, string url) => new(name, root, path, url, false, null);
      /// <summary>
      /// Create an instance of a test data folder
      /// </summary>
      /// <param name="name">Name of the object</param>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="builder">Optional custom builder</param>
      public static TestData Folder(string name, string root, string path, Builder builder = null) => new(name, root, path, null, true, builder);
      /// <summary>
      /// Create an instance of a test data folder
      /// </summary>
      /// <param name="name">Name of the object</param>
      /// <param name="root">Root of the path on disk</param>
      /// <param name="path">Path relative to the root</param>
      /// <param name="url">Url for download</param>
      public static TestData Folder(string name, string root, string path, string url) => new(name, root, path, url, true, null);
      /// <summary>
      /// Download/extract and get the path on disk
      /// </summary>
      /// <returns>The path of the test object</returns>
      public string Get()
      {
         // Check for existence
         var fullPath = IOPath.GetFullPath(IOPath.Combine(root, path));
         if ((isFolder && Directory.Exists(fullPath)) || (!isFolder && IOFile.Exists(fullPath)))
            return fullPath;
         // Custom build of test data
         if (builder != null)
            builder(fullPath);
         else {
            Assert.NotNull(url);
            var extract = isFolder || !string.IsNullOrEmpty(IOPath.GetDirectoryName(path));
            if (extract && IOPath.GetExtension(url).ToLower() == ".gz") {
               var tmpGZip = IOPath.Combine(IOPath.GetTempPath(), Guid.NewGuid().ToString());
               var tmpTar = IOPath.Combine(IOPath.GetTempPath(), Guid.NewGuid().ToString());
               try {
                  var webClient = new WebClient();
                  webClient.DownloadFile(url, tmpGZip);
                  GZip.Decompress(IOFile.OpenRead(tmpGZip), IOFile.Create(tmpTar), true);
                  using var tarStream = IOFile.OpenRead(tmpTar);
                  using var tar = TarArchive.CreateInputTarArchive(tarStream, Encoding.UTF8);
                  if (!Directory.Exists(root))
                     Directory.CreateDirectory(root);
                  tar.ExtractContents(root);
               }
               finally {
                  IOFile.Delete(tmpGZip);
                  IOFile.Delete(tmpTar);
               }
            }
            else if (extract && IOPath.GetExtension(url).ToLower() == ".zip") {
               var tmpZip = IOPath.Combine(IOPath.GetTempPath(), Guid.NewGuid().ToString());
               try {
                  var webClient = new WebClient();
                  webClient.DownloadFile(url, tmpZip);
                  if (!Directory.Exists(root))
                     Directory.CreateDirectory(root);
                  ZipFile.ExtractToDirectory(tmpZip, root, true);
               }
               finally {
                  IOFile.Delete(tmpZip);
               }
            }
            else {
               if (!Directory.Exists(root))
                  Directory.CreateDirectory(root);
               var webClient = new WebClient();
               webClient.DownloadFile(url, fullPath);
            }
         }
         Assert.True(isFolder || IOFile.Exists(fullPath));
         Assert.True(!isFolder || Directory.Exists(fullPath));
         return fullPath;
      }
      #endregion
   }
}
