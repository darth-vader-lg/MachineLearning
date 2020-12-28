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
   public class TextDataOptions
   {
      #region class Column
      [Serializable]
      public sealed class Column
      {
         #region Properties
         public string Name;
         public Range[] Source;
         public KeyCount KeyCount;
         public DataKind DataKind { get; set; }
         #endregion
         #region Methods
         public Column() : this(new TextLoader.Column()) { }
         public Column(TextLoader.Column col) => Fill(col);
         public Column(string name, DataKind dataKind, int index) => Fill(new TextLoader.Column(name, (Microsoft.ML.Data.DataKind)dataKind, index));
         public Column(string name, DataKind dataKind, int minIndex, int maxIndex) => Fill(new TextLoader.Column(name, (Microsoft.ML.Data.DataKind)dataKind, minIndex, maxIndex));
         public Column(string name, DataKind dataKind, Range[] source, KeyCount keyCount = null)
         {
            Fill(new TextLoader.Column(name, (Microsoft.ML.Data.DataKind)dataKind, source == default ? default(TextLoader.Range[]) : (from r in source select (TextLoader.Range)r).ToArray(), keyCount == default ? default : KeyCount));
         }
         private void Fill(TextLoader.Column col)
         {
            Name = col.Name;
            Source = col.Source == default ? default(Range[]) : (from r in col.Source select (Range)r).ToArray();
            KeyCount = col.KeyCount == default ? default(KeyCount) : col.KeyCount;
            DataKind = (DataKind)col.DataKind;
         }
         public static implicit operator TextLoader.Column(Column col)
         {
            return new TextLoader.Column
            {
               Name = col.Name,
               Source = col.Source == default ? default(TextLoader.Range[]) : (from r in col.Source select (TextLoader.Range)r).ToArray(),
               KeyCount = col.KeyCount == default ? default(Microsoft.ML.Data.KeyCount) : col.KeyCount,
               DataKind = (Microsoft.ML.Data.DataKind)col.DataKind,
            };
         }
         public static implicit operator Column(TextLoader.Column col) => new Column(col);
         #endregion
      }
      #endregion
      #region enum DataKind
      [Serializable]
      public enum DataKind
      {
         SByte = Microsoft.ML.Data.DataKind.SByte,
         Byte = Microsoft.ML.Data.DataKind.Byte,
         Int16 = Microsoft.ML.Data.DataKind.Int16,
         UInt16 = Microsoft.ML.Data.DataKind.UInt16,
         Int32 = Microsoft.ML.Data.DataKind.Int32,
         UInt32 = Microsoft.ML.Data.DataKind.UInt32,
         Int64 = Microsoft.ML.Data.DataKind.Int64,
         UInt64 = Microsoft.ML.Data.DataKind.UInt64,
         Single = Microsoft.ML.Data.DataKind.Single,
         Double = Microsoft.ML.Data.DataKind.Double,
         String = Microsoft.ML.Data.DataKind.String,
         Boolean = Microsoft.ML.Data.DataKind.Boolean,
         TimeSpan = Microsoft.ML.Data.DataKind.TimeSpan,
         DateTime = Microsoft.ML.Data.DataKind.DateTime,
         DateTimeOffset = Microsoft.ML.Data.DataKind.DateTimeOffset
      }
      #endregion
      #region class KeyCount
      [Serializable]
      public sealed class KeyCount
      {
         #region Properties
         public ulong? Count;
         #endregion
         #region Methods
         public KeyCount() : this(new Microsoft.ML.Data.KeyCount()) { }
         public KeyCount(Microsoft.ML.Data.KeyCount keyCount) => Fill(keyCount);
         public KeyCount(ulong count) : this(new Microsoft.ML.Data.KeyCount(count)) { }
         private void Fill(Microsoft.ML.Data.KeyCount keyCount) => Count = keyCount.Count;
         public static implicit operator Microsoft.ML.Data.KeyCount(KeyCount keyCount) => new Microsoft.ML.Data.KeyCount { Count = keyCount.Count };
         public static implicit operator KeyCount(Microsoft.ML.Data.KeyCount keyCount) => new KeyCount(keyCount);
         #endregion
      }
      #endregion
      #region class Range
      [Serializable]
      public sealed class Range
      {
         #region Properties
         public int Min;
         public int? Max;
         public bool AutoEnd;
         public bool VariableEnd;
         public bool AllOther;
         public bool ForceVector;
         #endregion
         #region Methods
         public Range() : this(new TextLoader.Range()) { }
         public Range(TextLoader.Range range) => Fill(range);
         public Range(int index) : this(new TextLoader.Range(index)) { }
         public Range(int min, int? max) : this(new TextLoader.Range(min, max)) { }
         private void Fill(TextLoader.Range range)
         {
            Min = range.Min;
            Max = range.Max;
            AutoEnd = range.AutoEnd;
            VariableEnd = range.VariableEnd;
            AllOther = range.AllOther;
            ForceVector = range.ForceVector;
         }
         public static implicit operator TextLoader.Range(Range range)
         {
            return new TextLoader.Range
            {
               Min = range.Min,
               Max = range.Max,
               AutoEnd = range.AutoEnd,
               VariableEnd = range.VariableEnd,
               AllOther = range.AllOther,
               ForceVector = range.ForceVector,
            };
         }
         public static implicit operator Range(TextLoader.Range range) => new Range(range);
         #endregion
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
      public TextDataOptions() : this(new TextLoader.Options()) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      public TextDataOptions(TextLoader.Options opt) => Fill(opt);
      /// <summary>
      /// Riempe con il contenuto di una TextDataOptions
      /// </summary>
      /// <param name="opt">Sorgente</param>
      private void Fill(TextLoader.Options opt)
      {
         AllowQuoting = opt.AllowQuoting;
         AllowSparse = opt.AllowSparse;
         InputSize = opt.InputSize;
         Separators = opt.Separators;
         DecimalMarker = opt.DecimalMarker;
         Columns = opt.Columns == default ? default(Column[]) : (from c in opt.Columns select (Column)c).ToArray();
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
      public static implicit operator TextLoader.Options(TextDataOptions opt)
      {
         return new TextLoader.Options()
         {
            AllowQuoting = opt.AllowQuoting,
            AllowSparse = opt.AllowSparse,
            InputSize = opt.InputSize,
            Separators = opt.Separators,
            DecimalMarker = opt.DecimalMarker,
            Columns = opt.Columns == default ? default(TextLoader.Column[]) : (from c in opt.Columns select (TextLoader.Column)c).ToArray(),
            TrimWhitespace = opt.TrimWhitespace,
            HasHeader = opt.HasHeader,
            UseThreads = opt.UseThreads,
            ReadMultilines = opt.ReadMultilines,
            HeaderFile = opt.HeaderFile,
            MaxRows = opt.MaxRows,
            EscapeChar = opt.EscapeChar,
            MissingRealsAsNaNs = opt.MissingRealsAsNaNs,
         };
      }
      /// <summary>
      /// Operatore di conversione da TextLoader.Options a TextLoaderOptions
      /// </summary>
      /// <param name="opt">TextLoader.Options</param>
      public static implicit operator TextDataOptions(TextLoader.Options opt) => new TextDataOptions(opt);
      #endregion
   }
}
