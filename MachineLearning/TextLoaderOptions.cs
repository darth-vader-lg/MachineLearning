using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Classe di wrapping delle TextLoader.Options di ML.NET che non sono serializzabili
   /// </summary>
   [Serializable]
   public class TextLoaderOptions
   {
      #region class Column
      [Serializable]
      public sealed class Column
      {
         #region Properties
         public string Name;
         public Range[] Source;
         public KeyCount KeyCount;
         public byte DataKind { get; set; }
         #endregion
      }
      #endregion
      #region class KeyCount
      [Serializable]
      public sealed class KeyCount
      {
         public ulong? Count;
      }
      #endregion
      #region class Range
      [Serializable]
      public sealed class Range
      {
         public int Min;
         public int? Max;
         public bool AutoEnd;
         public bool VariableEnd;
         public bool AllOther;
         public bool ForceVector;
      }
      #endregion
      #region Properties
      public bool AllowQuoting;
      public bool AllowSparse;
      public int? InputSize;
      public char[] Separators;
      public char DecimalMarker;
      public Column[] Columns;
      public bool TrimWhitespace;
      public bool HasHeader;
      public bool UseThreads;
      public bool ReadMultilines;
      public string HeaderFile;
      public long? MaxRows;
      public char EscapeChar;
      public bool MissingRealsAsNaNs;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public TextLoaderOptions() : this(columns: default) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="columns"></param>
      /// <param name="separator"></param>
      /// <param name="labelColumnName"></param>
      /// <param name="allowQuoting"></param>
      public TextLoaderOptions(IEnumerable<string> columns, char separator = ',', string labelColumnName = "Label", bool allowQuoting = true)
      {
         Fill(new TextLoader.Options
         {
            AllowQuoting = allowQuoting,
            AllowSparse = false,
            Separators = new[] { separator },
            Columns = columns != default ?
            columns.Select((c, i) => new TextLoader.Column(c == labelColumnName ? "Label" : c, DataKind.String, i)).ToArray() :
            new[]
            {
               new TextLoader.Column("Label", DataKind.String, 0),
               new TextLoader.Column("Text", DataKind.String, 1),
            }
         });
      }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="opt"></param>
      public TextLoaderOptions(TextLoader.Options opt) => Fill(opt);
      /// <summary>
      /// Riempe conil contenuto di una TextLoader.Opions
      /// </summary>
      /// <param name="opt">Sorgente</param>
      private void Fill(TextLoader.Options opt)
      {
         AllowQuoting = opt.AllowQuoting;
         AllowSparse = opt.AllowSparse;
         InputSize = opt.InputSize;
         Separators = opt.Separators;
         Columns =
            opt.Columns == default ? default :
            (from c in opt.Columns
             select new Column
             {
                DataKind = (byte)c.DataKind,
                KeyCount = c.KeyCount == default ? default :
                new KeyCount
                {
                   Count = c.KeyCount.Count
                },
                Name = c.Name,
                Source = c.Source == default ? default :
                (from s in c.Source
                 select new Range
                 {
                    AllOther = s.AllOther,
                    AutoEnd = s.AutoEnd,
                    ForceVector = s.ForceVector,
                    Max = s.Max,
                    Min = s.Min,
                    VariableEnd = s.VariableEnd
                 }).ToArray()
             }).ToArray();
         TrimWhitespace = opt.TrimWhitespace;
         HasHeader = opt.HasHeader;
         UseThreads = opt.UseThreads;
         ReadMultilines = opt.ReadMultilines;
         HeaderFile = opt.HeaderFile;
         MaxRows = opt.MaxRows;
         EscapeChar = opt.EscapeChar;
         MissingRealsAsNaNs = opt.MissingRealsAsNaNs;
      }
      /// <summary>
      /// Operatore di conversione a TextLoader.Options
      /// </summary>
      /// <param name="opt">TextLoaderOptions</param>
      public static implicit operator TextLoader.Options(TextLoaderOptions opt) =>
         new TextLoader.Options()
         {
            AllowQuoting = opt.AllowQuoting,
            AllowSparse = opt.AllowSparse,
            InputSize = opt.InputSize,
            Separators = opt.Separators,
            Columns =
            opt.Columns == default ? default :
            (from c in opt.Columns
             select new TextLoader.Column
             {
                DataKind = (DataKind)c.DataKind,
                KeyCount = c.KeyCount == default ? default:
                new Microsoft.ML.Data.KeyCount
                {
                   Count = c.KeyCount.Count
                },
                Name = c.Name,
                Source = c.Source == default ? default :
                (from s in c.Source
                 select new TextLoader.Range
                 {
                    AllOther = s.AllOther,
                    AutoEnd = s.AutoEnd,
                    ForceVector = s.ForceVector,
                    Max = s.Max,
                    Min = s.Min,
                    VariableEnd = s.VariableEnd
                 }).ToArray()
             }).ToArray(),
            TrimWhitespace = opt.TrimWhitespace,
            HasHeader = opt.HasHeader,
            UseThreads = opt.UseThreads,
            ReadMultilines = opt.ReadMultilines,
            HeaderFile = opt.HeaderFile,
            MaxRows = opt.MaxRows,
            EscapeChar = opt.EscapeChar,
            MissingRealsAsNaNs = opt.MissingRealsAsNaNs,
         };
      /// <summary>
      /// Operatore di conversione da TextLoader.Options a TextLoaderOptions
      /// </summary>
      /// <param name="opt">TextLoader.Options</param>
      public static implicit operator TextLoaderOptions(TextLoader.Options opt) => new TextLoaderOptions(opt);
      #endregion
   }
}
