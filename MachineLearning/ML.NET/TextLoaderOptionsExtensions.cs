using MachineLearning;
using System.Linq;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Estensioni alla classe TextLoader.Options
   /// </summary>
   public static class TextLoaderOptionsExtensions
   {
      #region Methods
      /// <summary>
      /// Splitta i dati di una riga in base alle colonne
      /// </summary>
      /// <param name="row">Riga di dati</param>
      /// <returns>I dati splittati</returns>
      public static string[] SplitData(this TextDataOptions options, string row) => SplitData((TextLoader.Options)options, row);
      /// <summary>
      /// Splitta i dati di una riga in base alle colonne
      /// </summary>
      /// <param name="row">Riga di dati</param>
      /// <returns>I dati splittati</returns>
      public static string[] SplitData(this TextLoader.Options options, string row)
      {
         var expr = @"(?<match>\w+)|\""(?<match>[\w\s";
         foreach (var separator in options.Separators != default && options.Separators.Length > 0 ? options.Separators : new[] { ',' })
            expr += new string(separator, 1);
         expr += @"]*)""";
         return Regex.Matches(row, expr).Cast<Match>().Select(m => m.Groups["match"].Value).ToArray();
      }
      #endregion
   }
}
